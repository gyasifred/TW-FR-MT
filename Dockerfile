FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 GRADIO_ANALYTICS_ENABLED=False
WORKDIR /app
RUN useradd --create-home --uid 1000 appuser
COPY --chown=appuser:appuser pyproject.toml README.md LICENSE ./
COPY --chown=appuser:appuser twfr_mt ./twfr_mt
COPY --chown=appuser:appuser configs ./configs
RUN pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6,<3' && pip install '.[inference,app]'
USER appuser
ENV TWFR_MODEL_ROOT=/models PORT=7860
EXPOSE 7860
CMD ["twfr-mt", "serve", "--host", "0.0.0.0"]
