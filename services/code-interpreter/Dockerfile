# Pinned by digest, not by `:main`. That tag moves, and this is the container
# that runs user-supplied code — a base that changes under a rebuild changes
# the sandbox. The vLLM image is pinned the same way, for the same reason:
# a moving base once relocated the tool-parser registry and left containers
# healthy with tool calls silently unparsed.
#
# To move it: docker buildx imagetools inspect ghcr.io/usnavy13/librecodeinterpreter:main
# and record the manifest-list digest here, deliberately.
FROM ghcr.io/usnavy13/librecodeinterpreter:main@sha256:670f50425a66ed9b410f5ae1209b526f6254abe269ff2b1c479485923bb5e064

COPY fonts/NanumGothic-Regular.ttf /usr/share/fonts/truetype/NanumGothic-Regular.ttf

# Make NanumGothic the matplotlib default so Korean chart labels from
# execute_code do not render as tofu. Existing font.* keys are dropped first and
# added once, which avoids Duplicate/Bad key warnings.
RUN MPLRC="$(python3 -c 'import matplotlib; print(matplotlib.matplotlib_fname())')" && \
    sed -i -E '/^#?[[:space:]]*font\.family[[:space:]]*:/d; /^#?[[:space:]]*font\.sans-serif[[:space:]]*:/d; /^#?[[:space:]]*axes\.unicode_minus[[:space:]]*:/d' "$MPLRC" && \
    printf 'font.family: sans-serif\nfont.sans-serif: NanumGothic, DejaVu Sans, sans-serif\naxes.unicode_minus: False\n' >> "$MPLRC" && \
    fc-cache -f >/dev/null 2>&1 || true

RUN rm -rf /root/.cache/matplotlib /home/*/.cache/matplotlib 2>/dev/null || true
