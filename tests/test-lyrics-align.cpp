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
        byte_level_encode(&tok, "[verse]\n"),
        byte_level_encode(&tok, "hello\n"),
        byte_level_encode(&tok, "world"),
    };
    tok.n_vocab = (int) tok.id_to_str.size();

    const int n_tokens = 3;
    const int n_frames = 9;
    std::vector<float> scores((size_t) MM3_ALIGN_N_HEADS * n_tokens * n_frames, 0.001f);
    for (int head = 0; head < MM3_ALIGN_N_HEADS; head++) {
        for (int frame = 0; frame < n_frames; frame++) {
            const int token = frame / 3;
            scores[((size_t) head * n_tokens + (size_t) token) * n_frames + (size_t) frame] = 1.0f;
        }
    }
    const std::string lrc = mm3_align_build_lrc(scores, { 0, 1, 2 }, tok, n_frames,
                                                 (float) n_frames / 25.0f);
    if (!expect(lrc == "[00:00.82]hello\n[00:00.94]world\n", "unexpected LRC text or timestamps")) {
        fprintf(stderr, "[Test-LRC] got:\n%s", lrc.c_str());
        return 1;
    }

    fprintf(stderr, "[Test-LRC] OK\n");
    return 0;
}
