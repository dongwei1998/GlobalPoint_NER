docker build --network=host -t registry-svc:25000/library/GlobalPointer_model:v1.0.0 .
docker push registry-svc:25000/library/GlobalPointer_model:v1.0.0
FROM registry-svc:25000/library/ubuntu_py3.9.8_tf2.5.0_cuda11:v1.0.3