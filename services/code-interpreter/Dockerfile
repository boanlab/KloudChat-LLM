FROM ghcr.io/usnavy13/librecodeinterpreter:main

COPY fonts/NanumGothic-Regular.ttf /usr/share/fonts/truetype/NanumGothic-Regular.ttf

# Make NanumGothic the matplotlib default so Korean chart labels from
# execute_code do not render as tofu. Existing font.* keys are dropped first and
# added once, which avoids Duplicate/Bad key warnings.
RUN MPLRC="$(python3 -c 'import matplotlib; print(matplotlib.matplotlib_fname())')" && \
    sed -i -E '/^#?[[:space:]]*font\.family[[:space:]]*:/d; /^#?[[:space:]]*font\.sans-serif[[:space:]]*:/d; /^#?[[:space:]]*axes\.unicode_minus[[:space:]]*:/d' "$MPLRC" && \
    printf 'font.family: sans-serif\nfont.sans-serif: NanumGothic, DejaVu Sans, sans-serif\naxes.unicode_minus: False\n' >> "$MPLRC" && \
    fc-cache -f >/dev/null 2>&1 || true

RUN rm -rf /root/.cache/matplotlib /home/*/.cache/matplotlib 2>/dev/null || true
