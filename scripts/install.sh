pip3 install -U pip && pip3 install -r requirements.txt
pip3 install -v -e .


cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
