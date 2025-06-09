FROM nvcr.io/nvidia/tritonserver:23.12-py3

# Install required Python packages
RUN pip install joblib scikit-learn numpy pandas

# Models will be mounted as volume from docker-compose.yaml
# No need to copy models during build 