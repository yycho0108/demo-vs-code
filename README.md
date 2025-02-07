
## Demo Vs Code

### Install

First, check your python version with `python3 --version` and edit `requirements.txt` to match the version.
You can check the `.whl` urls from the [OMPL releases](https://github.com/ompl/ompl/releases/).
```
ompl @ https://github.com/ompl/ompl/releases/download/prerelease/ompl-1.6.0-cp38-cp38-manylinux_2_28_x86_64.whl
                                                                            ^^^^^^^^^ this is for python3.8
```


```bash
# [Optional] use virtualenv
# python3 -m venv avenv 
# source venv/bin/activate
python3 -m pip install -r requirements.txt
git submodule update --init --recursive
```

