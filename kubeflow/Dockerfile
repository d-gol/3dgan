ARG IMAGE_TYPE="gpu"
FROM gitlab-registry.cern.ch/ai-ml/images/tensorflow-2.1.0-notebook-gpu:1.0.0-2
USER root

ENV NB_PREFIX /
    
ENV SHELL /bin/bash

COPY requirements/requirements.txt /tmp/requirements.txt
COPY krb5cc_1000 /tmp/krb5cc_1000
COPY krb5cc_1000 /tmp/krb5cc_0
COPY gan.py /
ADD gan.py /opt/gan.py

RUN apt-get -qq update && \
/usr/bin/python3 -m pip install --upgrade pip && \
pip3 install -r /tmp/requirements.txt && \
chmod 600 /tmp/krb5cc_1000 && \
chown jovyan /tmp/krb5cc_1000 && \
chmod 600 /tmp/krb5cc_0 && \
echo "jovyan ALL=(ALL:ALL) NOPASSWD:ALL" > /etc/sudoers.d/jovyan && \
chmod +x /opt/gan.py

#ENTRYPOINT ["/usr/bin/python3"]
#CMD ["/opt/GPU_3DGAN.py"]
