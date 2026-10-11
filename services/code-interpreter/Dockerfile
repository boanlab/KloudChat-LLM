# Pinned by digest: `:main` moves, and this container runs user code.
# To move it: docker buildx imagetools inspect ghcr.io/usnavy13/librecodeinterpreter:main
FROM ghcr.io/usnavy13/librecodeinterpreter:main@sha256:670f50425a66ed9b410f5ae1209b526f6254abe269ff2b1c479485923bb5e064

COPY fonts/NanumGothic-Regular.ttf /usr/share/fonts/truetype/NanumGothic-Regular.ttf

# NanumGothic as the matplotlib default, for Korean chart labels.
RUN MPLRC="$(python3 -c 'import matplotlib; print(matplotlib.matplotlib_fname())')" && \
    sed -i -E '/^#?[[:space:]]*font\.family[[:space:]]*:/d; /^#?[[:space:]]*font\.sans-serif[[:space:]]*:/d; /^#?[[:space:]]*axes\.unicode_minus[[:space:]]*:/d' "$MPLRC" && \
    printf 'font.family: sans-serif\nfont.sans-serif: NanumGothic, DejaVu Sans, sans-serif\naxes.unicode_minus: False\n' >> "$MPLRC" && \
    fc-cache -f >/dev/null 2>&1 || true

# PDF libraries for file skills. The sandbox has no network at runtime, so they are baked in here,
# into the system interpreter the nsjail sandbox runs (/usr/bin/python3 -> /usr/local/lib/python3.12/dist-packages).
# The base image ships the old `pdfminer` (20191125) on top of pdfminer.six, which breaks
# `pdfminer.high_level` (ImportError HOCRConverter): drop it, then restore pdfminer.six's files.
RUN pip uninstall -y --break-system-packages pdfminer && \
    pip install --no-cache-dir --break-system-packages --force-reinstall --no-deps pdfminer.six==20260107 && \
    pip install --no-cache-dir --break-system-packages pdfplumber==0.11.10 pypdf==6.19.0 pypdfium2==5.14.0 && \
    python3 -c "import pypdf, pdfplumber, pdfminer.high_level, PyPDF2, pandas, openpyxl"

# Per-sandbox memory limit (cgroup v2), wired up in docker-compose.yml with
# `command: [cgroup-init, ...]` and `NSJAIL_BINARY=/usr/local/bin/nsjail-memcg`.
COPY --chmod=755 bin/cgroup-init bin/nsjail-memcg /usr/local/bin/

RUN rm -rf /root/.cache/matplotlib /home/*/.cache/matplotlib 2>/dev/null || true
