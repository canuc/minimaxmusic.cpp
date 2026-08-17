#pragma once
// lyrics-align.h: native MiniMax Music3 lyric-to-frame alignment and LRC output
//
// Derived from HOT-Step-CPP's minimax/mm3-align.h at commit
// a5c8ed3d6c8e47c116557ad66b14452049822e2b. HOT-Step-CPP and this
// project are MIT licensed; see THIRD_PARTY_NOTICES.md.
//
// Every 25 Hz autoregressive decode query attends to the lyric tokens in the
// prompt. The three empirically selected heads below form a
// [head][lyric token][audio frame] matrix. We normalize and average those
// heads, find a legal reading-order path with monotonic DTW, then timestamp
// each non-structural lyric line at its first token onset.

#include "bpe.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct MM3AlignHead {
    int layer;
    int head;
};

static const MM3AlignHead MM3_ALIGN_HEADS[] = {
    { 12, 27 },
    { 19,  7 },
    { 24, 29 },
};
static const int MM3_ALIGN_N_HEADS = (int) (sizeof(MM3_ALIGN_HEADS) / sizeof(MM3_ALIGN_HEADS[0]));

// LM attention leads the audible vocal by about 0.6--0.8 seconds in the
// HOT-Step measurements. Apply the midpoint to every emitted line start.
static const float MM3_ALIGN_LEAD_SEC = 0.70f;

static inline int mm3_align_head_for_layer(int layer) {
    for (int i = 0; i < MM3_ALIGN_N_HEADS; i++) {
        if (MM3_ALIGN_HEADS[i].layer == layer) {
            return i;
        }
    }
    return -1;
}

// Decode each byte-level BPE token into only the bytes it contributes to the
// accumulated lyric string.
static void mm3_align_token_texts(const BPETokenizer & bpe,
                                  const std::vector<int> & ids,
                                  std::vector<std::string> * out) {
    out->assign(ids.size(), std::string());
    std::string prev_full;
    std::string full;
    for (size_t i = 0; i < ids.size(); i++) {
        const int id = ids[i];
        if (id >= 0 && id < bpe.n_vocab) {
            const std::string & encoded = bpe.id_to_str[(size_t) id];
            for (size_t p = 0; p < encoded.size();) {
                int       advance = 0;
                const int codepoint = utf8_codepoint(encoded.c_str() + p, &advance);
                bool      found = false;
                for (int byte = 0; byte < 256; byte++) {
                    int       byte_advance = 0;
                    const int byte_codepoint = utf8_codepoint(bpe.byte2str[byte].c_str(), &byte_advance);
                    if (byte_codepoint == codepoint && (int) bpe.byte2str[byte].size() == advance) {
                        full += (char) (unsigned char) byte;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    full += '?';
                }
                p += (size_t) (advance > 0 ? advance : 1);
            }
        }
        if (full.size() > prev_full.size()) {
            (*out)[i] = full.substr(prev_full.size());
        }
        prev_full = full;
    }
}

// Monotonic DTW over cost[token][frame]. The returned path maps each frame to
// one token; a step may stay on the token or advance by exactly one token.
static bool mm3_align_dtw(const std::vector<float> & cost,
                          int n_tokens,
                          int n_frames,
                          std::vector<int> * path) {
    if (n_tokens <= 0 || n_frames <= 0 || n_tokens > n_frames ||
        cost.size() != (size_t) n_tokens * (size_t) n_frames) {
        return false;
    }

    const float        inf = 1e30f;
    std::vector<float> acc((size_t) n_tokens * (size_t) n_frames, inf);
    std::vector<uint8_t> back((size_t) n_tokens * (size_t) n_frames, 0);

    acc[0] = cost[0];
    for (int frame = 1; frame < n_frames; frame++) {
        acc[(size_t) frame] = acc[(size_t) frame - 1] + cost[(size_t) frame];
    }
    for (int token = 1; token < n_tokens; token++) {
        for (int frame = 1; frame < n_frames; frame++) {
            const size_t at = (size_t) token * (size_t) n_frames + (size_t) frame;
            const float  stay = acc[at - 1];
            const float  advance = acc[at - (size_t) n_frames - 1];
            if (stay <= advance) {
                acc[at] = stay + cost[at];
            } else {
                acc[at]  = advance + cost[at];
                back[at] = 1;
            }
        }
    }
    if (!std::isfinite(acc.back()) || acc.back() >= inf) {
        return false;
    }

    path->assign((size_t) n_frames, 0);
    int token = n_tokens - 1;
    for (int frame = n_frames - 1; frame > 0; frame--) {
        (*path)[(size_t) frame] = token;
        if (token > 0 && back[(size_t) token * (size_t) n_frames + (size_t) frame]) {
            token--;
        }
    }
    (*path)[0] = token;
    return token == 0;
}

static std::string mm3_align_stamp(float seconds) {
    if (seconds < 0.0f) {
        seconds = 0.0f;
    }
    const int centiseconds = (int) (seconds * 100.0f + 0.5f);
    char      buf[32];
    snprintf(buf, sizeof(buf), "[%02d:%02d.%02d]", centiseconds / 6000,
             (centiseconds / 100) % 60, centiseconds % 100);
    return buf;
}

// scores layout: [MM3_ALIGN_N_HEADS][lyric token][audio frame].
// Returns empty on invalid or insufficient evidence so callers can fall back
// to an ASR aligner without failing music generation.
static std::string mm3_align_build_lrc(const std::vector<float> & scores,
                                       const std::vector<int> & lyric_ids,
                                       const BPETokenizer & bpe,
                                       int n_frames,
                                       float duration_seconds) {
    const int n_tokens = (int) lyric_ids.size();
    const size_t expected = (size_t) MM3_ALIGN_N_HEADS * (size_t) n_tokens * (size_t) n_frames;
    if (n_tokens <= 1 || n_frames <= 1 || n_tokens > n_frames || duration_seconds <= 0.0f ||
        scores.size() != expected) {
        return std::string();
    }

    std::vector<std::string> token_texts;
    mm3_align_token_texts(bpe, lyric_ids, &token_texts);

    // Normalize each head over lyric tokens for each frame before averaging,
    // preventing one high-magnitude head from drowning out the others.
    std::vector<float> average((size_t) n_tokens * (size_t) n_frames, 0.0f);
    for (int head = 0; head < MM3_ALIGN_N_HEADS; head++) {
        const float * source = scores.data() + (size_t) head * (size_t) n_tokens * (size_t) n_frames;
        for (int frame = 0; frame < n_frames; frame++) {
            float sum = 0.0f;
            for (int token = 0; token < n_tokens; token++) {
                const float value = source[(size_t) token * (size_t) n_frames + (size_t) frame];
                if (std::isfinite(value) && value > 0.0f) {
                    sum += value;
                }
            }
            if (sum <= 1e-9f) {
                continue;
            }
            const float inverse = 1.0f / sum;
            for (int token = 0; token < n_tokens; token++) {
                const float value = source[(size_t) token * (size_t) n_frames + (size_t) frame];
                if (std::isfinite(value) && value > 0.0f) {
                    average[(size_t) token * (size_t) n_frames + (size_t) frame] += value * inverse;
                }
            }
        }
    }

    std::vector<float> cost(average.size());
    for (size_t i = 0; i < average.size(); i++) {
        cost[i] = -logf(average[i] / (float) MM3_ALIGN_N_HEADS + 1e-9f);
    }

    std::vector<int> path;
    if (!mm3_align_dtw(cost, n_tokens, n_frames, &path)) {
        return std::string();
    }

    std::vector<int> onset((size_t) n_tokens, -1);
    for (int frame = 0; frame < n_frames; frame++) {
        const int token = path[(size_t) frame];
        if (token >= 0 && token < n_tokens && onset[(size_t) token] < 0) {
            onset[(size_t) token] = frame;
        }
    }

    const float seconds_per_frame = duration_seconds / (float) n_frames;
    std::string out;
    std::string line;
    int         line_first_token = 0;
    auto flush_line = [&]() {
        const size_t first = line.find_first_not_of(" \t\r\n");
        const size_t last  = line.find_last_not_of(" \t\r\n");
        if (first != std::string::npos && last != std::string::npos) {
            const std::string body = line.substr(first, last - first + 1);
            // The normalized prompt contains structural lines such as [start],
            // [verse], and [chorus]. They guide generation but are not lyrics.
            if (!body.empty() && body[0] != '[') {
                int frame = onset[(size_t) line_first_token];
                if (frame < 0) {
                    frame = 0;
                }
                out += mm3_align_stamp((float) frame * seconds_per_frame + MM3_ALIGN_LEAD_SEC);
                out += body;
                out += '\n';
            }
        }
        line.clear();
    };

    for (int token = 0; token < n_tokens; token++) {
        const std::string & text = token_texts[(size_t) token];
        for (char ch : text) {
            if (ch == '\n') {
                flush_line();
                line_first_token = token + 1 < n_tokens ? token + 1 : token;
            } else {
                if (line.empty()) {
                    line_first_token = token;
                }
                line += ch;
            }
        }
    }
    flush_line();
    return out;
}
