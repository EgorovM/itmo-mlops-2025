FROM nvcr.io/nvidia/tritonserver:23.12-py3

# Install required Python packages
RUN pip install joblib scikit-learn numpy pandas

# Copy the model repository
COPY models /models

# Start triton server
CMD ["tritonserver", "--model-repository=/models", "--strict-model-config=false"] 