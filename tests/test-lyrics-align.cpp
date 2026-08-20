// Pure CPU checks for the optional MiniMax Music3 attention-to-LRC path.

#include "lyrics-align.h"

#include <cstdio>

static bool expect(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "[Test-LRC] FAIL: %s\n", message);
        return false;
    }
    return true;
}

int main() {
    // Three-token diagonal path, two frames per token.
    std::vector<float> cost(18, 20.0f);
    for (int frame = 0; frame < 6; frame++) {
        cost[(size_t) (frame / 2) * 6 + (size_t) frame] = 0.0f;
    }
    std::vector<int> path;
    if (!expect(mm3_align_dtw(cost, 3, 6, &path), "DTW rejected a legal path") ||
        !expect(path == std::vector<int>({ 0, 0, 1, 1, 2, 2 }), "DTW chose the wrong path") ||
        !expect(!mm3_align_dtw(cost, 7, 6, &path), "DTW accepted more tokens than frames")) {
        return 1;
    }

    BPETokenizer tok = {};
    build_byte_encoder(tok.byte2str);
    tok.id_to_str = {
        byte_level_encode(&tok, "[verse]\n"), byte_level_encode(&tok, "hel"),   byte_level_encode(&tok, "lo "),
        byte_level_encode(&tok, "little\n"),  byte_level_encode(&tok, "world"),
    };
    tok.n_vocab = (int) tok.id_to_str.size();

    const int          n_tokens = 5;
    const int          n_frames = 15;
    std::vector<float> scores((size_t) MM3_ALIGN_N_HEADS * n_tokens * n_frames, 0.001f);
    for (int head = 0; head < MM3_ALIGN_N_HEADS; head++) {
        for (int frame = 0; frame < n_frames; frame++) {
            const int token                                                                 = frame / 3;
            scores[((size_t) head * n_tokens + (size_t) token) * n_frames + (size_t) frame] = 1.0f;
        }
    }
    const MM3AlignOutput output =
        mm3_align_build_outputs(scores, { 0, 1, 2, 3, 4 }, tok, n_frames, (float) n_frames / 25.0f);
    if (!expect(output.lrc == "[00:00.82]hello little\n[00:01.18]world\n", "unexpected LRC text or timestamps")) {
        fprintf(stderr, "[Test-LRC] got:\n%s", output.lrc.c_str());
        return 1;
    }
    const std::string expected_words =
        "{\"version\":1,\"method\":\"minimax-lyric-attention-dtw\",\"frameDurationS\":0.040,\"words\":["
        "{\"word\":\"hello\",\"startS\":0.820,\"endS\":1.060,\"startTokenIndex\":1,\"endTokenIndex\":2,"
        "\"startFrame\":3,\"endFrame\":9},"
        "{\"word\":\"little\",\"startS\":1.060,\"endS\":1.180,\"startTokenIndex\":3,\"endTokenIndex\":3,"
        "\"startFrame\":9,\"endFrame\":12},"
        "{\"word\":\"world\",\"startS\":1.180,\"endS\":1.300,\"startTokenIndex\":4,\"endTokenIndex\":4,"
        "\"startFrame\":12,\"endFrame\":15}]}";
    if (!expect(output.word_spans_json == expected_words, "unexpected token-derived word spans")) {
        fprintf(stderr, "[Test-LRC] word spans got:\n%s\n", output.word_spans_json.c_str());
        return 1;
    }

    fprintf(stderr, "[Test-LRC] OK\n");
    return 0;
}
