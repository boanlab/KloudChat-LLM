# Pinned by digest: `:main` moves, and this container runs user code.
# To move it: docker buildx imagetools inspect ghcr.io/usnavy13/librecodeinterpreter:main
FROM ghcr.io/usnavy13/librecodeinterpreter:main@sha256:670f50425a66ed9b410f5ae1209b526f6254abe269ff2b1c479485923bb5e064

COPY fonts/NanumGothic-Regular.ttf /usr/share/fonts/truetype/NanumGothic-Regular.ttf

# NanumGothic as the matplotlib default, for Korean chart labels.
RUN MPLRC="$(python3 -c 'import matplotlib; print(matplotlib.matplotlib_fname())')" && \
    sed -i -E '/^#?[[:space:]]*font\.family[[:space:]]*:/d; /^#?[[:space:]]*font\.sans-serif[[:space:]]*:/d; /^#?[[:space:]]*axes\.unicode_minus[[:space:]]*:/d' "$MPLRC" && \
    printf 'font.family: sans-serif\nfont.sans-serif: NanumGothic, DejaVu Sans, sans-serif\naxes.unicode_minus: False\n' >> "$MPLRC" && \
    fc-cache -f >/dev/null 2>&1 || true

RUN rm -rf /root/.cache/matplotlib /home/*/.cache/matplotlib 2>/dev/null || true
