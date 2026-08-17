// Request contract checks that do not require model files.

#include "request.h"

#include <cstdio>

int main() {
    MM3Request request;
    request_init(&request);
    if (request.get_lrc) {
        fprintf(stderr, "[Test-Request] FAIL: get_lrc default is not false\n");
        return 1;
    }
    if (!request_parse_json(&request, "{\"caption\":\"c\",\"lyrics\":\"l\",\"get_lrc\":true}")) {
        fprintf(stderr, "[Test-Request] FAIL: parse failed\n");
        return 1;
    }
    if (!request.get_lrc) {
        fprintf(stderr, "[Test-Request] FAIL: get_lrc was not parsed\n");
        return 1;
    }
    const std::string json = request_to_json(&request, true);
    if (json.find("\"get_lrc\": true") == std::string::npos) {
        fprintf(stderr, "[Test-Request] FAIL: get_lrc was not serialized: %s\n", json.c_str());
        return 1;
    }
    fprintf(stderr, "[Test-Request] OK\n");
    return 0;
}
