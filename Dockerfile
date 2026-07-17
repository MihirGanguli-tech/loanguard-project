# Pull lambda runtime image
FROM public.ecr.aws/lambda/python:3.10 
#LightGBM needs c++ compiler
RUN yum install -y gcc gcc-c++ cmake make


#Install dependencies
COPY requirements_lambda.txt .
RUN pip install -r requirements_lambda.txt -t "${LAMBDA_TASK_ROOT}"

# Copy the app and src directories, need preprocess.py since raw applicant data is being processed
COPY app/ ${LAMBDA_TASK_ROOT}/app/
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY models/lightgbm.joblib ${LAMBDA_TASK_ROOT}/models/lightgbm.joblib

#define lambda function
CMD ["app.main.handler"]