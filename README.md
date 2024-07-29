# llama2-sft-chatbot

## License
This template is licensed under Apache 2.0 and contains the following components: 
* Llama2/Llama3.1 [License](https://ai.meta.com/llama/license/)
* MLFlow [Apache 2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
* ctranslate [MIT](https://github.com/OpenNMT/CTranslate2/blob/master/LICENSE)
* pytorch [Caffe 2](https://github.com/pytorch/pytorch/blob/main/LICENSE)
* Transformer [Apache 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE)
* peft [Apache 2.0](https://github.com/huggingface/peft/blob/main/LICENSE)
* SFTTrainer [Apache 2.0](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py)
* Unsloth [Apache 2.0](https://unsloth.ai/terms)

## About this project
In this project we demonstrate the use of a pre-trained Large Language Model (LLM) in Domino and the process of fine-tuning the model for a specific task using Llama2 or Llama3. In the python code provide, we will convert this model using `ctranslate2` to optimize its throughput and deploy it as a model API and app in Domino.

Fine-tuning a pre-trained LLM is a commonly used technique for solving NLP problems with machine learning. This is a typical transfer learning task where the final model is realised through a number of training phases:

1. The process typically begins with a pre-trained model, which is not task specific. This model is trained on a large corpora of unlabelled data 

2. The model undergoes a process of domain specific adaptive fine-tuning, which produces a new model with narrower focus or better alignment. This new model is better prepared to address domain-specific challenges as it is now closer to the expected distribution of the target data or responses the user expects. 

In this demo project we use the [mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k) dataset, which provides 1000 samples of the excellent [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) dataset, in two distinct notebooks processed to match Llama 3.1 or Llama 2's prompt format . This dataset is used in conjuction with [NousResearch/Llama-2-7b-chat-hf](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf) or [NousResearch/Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B), which we fine-tune for the purpose of building a conversational assistant.

The assets available in this project are:

*llama3_1_guanco.ipynb* - A notebook, illustrating the process of finetuning [NousResearch/Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B) on the [mlabonne/guanaco-llama2-1k] (https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k) dataset

*llama2_guanaco.ipynb* - A notebook, illustrating the process of  fine tuning [NousResearch/Llama-2-7b-chat-hf](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf) on the [mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k) dataset

*ct2_converter.ipynb* - A notebook that shows how to convert the fine tuned Huggingface model to an optimized `ctranslate2` model

*model.py* - A scoring function, which is used to deploy the fine-tuned model as a [Domino model API](https://docs.dominodatalab.com/en/latest/user_guide/8dbc91/host-models-as-rest-apis/)

*app_streaming.py* - A Python file that embeds the `ctranslate2` model and allows for users to interact with the model as a Streamlit app

*app.sh* - Launch instructions for the accompanying Streamlit app.

## Prerequisites

Github Personal Access Token (PAT) Authentication in Domino [Domino PAT Setup](https://docs.dominodatalab.com/en/latest/user_guide/314004/import-git-repositories/)

For Llama 3.1 notebook: 
A Hugging Face User Access Token is required to run the LLama3 notebook [Uster Access Token](https://huggingface.co/docs/hub/en/security-tokens)


## Set up instructions if interested in just running the Jupyter Notebooks 
### Environment Requirements 
Use a Domino Standard Enviornment during set up

### Hardware Requirements
Use a GPU Hardware Tier, this is required, reccomended 22GB RAM. 


## Set up instructions for Python File, Required if using the API and Streamlit application
## Model API calls

The **model.py** provides a scoring function with the following signature: `generate(prompt)`. To test it, you could use the following JSON payload:

```
{
  "data": {
    "prompt": "What can I do in Paris for a day?"
  }
}
```

**Please note that in order to run the model API you need `ctranslate2 3.17.1` which is not part of the environment base mentioned in the section below. You will have to add this dependency to the environment Dockerfile instructions.** 


This project requires the following [compute environments](https://docs.dominodatalab.com/en/latest/user_guide/f51038/environments/) to be present. Please ensure the "Automatically make compatible with Domino" checkbox is selected while creating the environment.

If you are interested in running just the Jupyter notebooks, use a 

### Hardware Requirements for Python File with API 

You also need to make sure that the hardware tier running the notebook or the fine-tuning script has sufficient resources. A GPU with >=16GB of VRAM is recommended. This project was tested on a `V100` with **16GB** VRAM. Also note that the model binary occupies ~ **28GB** on disc so please provision your workspace volume accordingly.

### Environment Requirements for Python File API 

`quay.io/domino/pre-release-environments:domino-llm-environment.main.latest`

**Pluggable Workspace Tools** 
```
jupyterlab:
  title: "JupyterLab"
  iconUrl: "/assets/images/workspace-logos/jupyterlab.svg"
  start: [ "/opt/domino/workspaces/jupyterlab/start" ]
  httpProxy:
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    port: 8888
    rewrite: false
    requireSubdomain: false
vscode:
 title: "vscode"
 iconUrl: "/assets/images/workspace-logos/vscode.svg"
 start: [ "/opt/domino/workspaces/vscode/start" ]
 httpProxy:
    port: 8888
    requireSubdomain: false
```

Please change the value in `start` according to your Domino version.

In case you want to create a custom environment for the project, you can do so by specifying the following when creating a custom environment

**Environment Base Python File API**

***custom base image :*** `nvcr.io/nvidia/pytorch:22.12-py3`

***Dockerfile instructions***
```
RUN pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
RUN pip install -q datasets bitsandbytes einops mlflow langchain langsmith openai textstat evaluate rapidfuzz tiktoken
RUN pip install --no-cache-dir Flask Flask-Compress Flask-Cors jsonify uWSGI streamlit "ctranslate2==3.17.1"
# RUN pip install streamlit-chat
RUN pip install -i https://test.pypi.org/simple/ streamlit-chat-domino 
RUN pip uninstall --yes transformer-engine
RUN pip uninstall -y apex

RUN pip uninstall --yes torch torchvision torchaudio

RUN pip install torch  --index-url https://download.pytorch.org/whl/cu118
```
