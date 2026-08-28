FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so edits to the app don't invalidate the pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY *.py — NOT just app.py. app.py loads monitoring_dashboard.py and
# research_dashboard.py dynamically from disk on every rerun and falls back to
# a "coming soon" placeholder when a file is missing. Copying only app.py
# therefore builds cleanly and starts fine, but silently ships a dashboard with
# two of its views permanently stubbed out.
COPY *.py ./
COPY static/ static/

# Populated at runtime by the init container in aa-infra
# (components/ocp-dashboard), which stages the parquet + map_data.json here.
ENV OCP_DATA_DIR=/data
RUN mkdir -p /data

EXPOSE 8501

# baseUrlPath must match the ingress path and the probes in the Deployment
# (/ocp-dashboard). CORS/XSRF are disabled because the app is served behind
# ingress-nginx on a subpath.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.baseUrlPath=/ocp-dashboard", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--server.headless=true"]
