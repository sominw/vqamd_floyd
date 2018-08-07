  # Visual Question Answering through Modal Dialogue

<p align="center">
  <img src="https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/vqa/1.jpeg?raw=true"/>
</p>

We’re already seeing incredible applications of object detection in our daily lives. One such interesting application is Visual Question Answering. It is a new and upcoming problem in Computer Vision where the data consists of open-ended questions about images. In order to answer these questions, an effective system would need to have an understanding of  “[vision, language and common-sense](https://dac.cs.vt.edu/research-project/visual-question-answering-vqa/).”

Before proceeding further, I would highly encourge you to quickly read the full VQA Post here. 

## Try it now on FloydHub

<p align="center">
<a href="https://floydhub.com/run">
    <img src="https://static.floydhub.com/button/button.svg" alt="Run">
</a>
</p>

Click this button to open a Workspace on FloydHub that will train this model.

Do remember to execute **`run_me_first_floyd.sh` inside a terminal everytime** you restart your workspace to install relevant dependencies. 
<!---
<p align="center">
  <img src="https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/vqa/9.png?raw=true"/>
</p>
-->
---

This post will first dig into the basic theory behind the Visual Question Answering task. Then, we’ll discuss and build two approaches to VQA: the “bag-of-words” and the “recurrent” model. Finally, we’ll provide a tutorial workflow for training your own models and setting up a REST API on FloydHub to start detecting objects in your own images.
The project code is in Python (Keras + TensorFlow). You can view my experiments directly on [FloydHub](https://www.floydhub.com/sominw/projects/vqa_floyd), as well as the code (along with the weight files and data) on [Github](https://github.com/sominwadhwa/vqamd_floyd).

<p align="center">
  <img src="https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/vqa/8.gif?raw=true"/>
</p>

Since I've already preprocessed the data & stored everything in a FloydHub [dataset](https://www.floydhub.com/sominw/datasets/vqa_data), here's what we're going to do - 

- Checkout the preprocessed data from the VQA Dataset.
- Build & train **two** `VQA` models using Keras & Tensorflow.
- Assess the models on the `VQA` validation sets.
- Run the model to generate some really cool predictions. 
---

## For Offline Execution

The following are a couple of instructions that must be gone through in order to execute different (or all) sections of this project. You will need a **NVIDIA GPU** to train these models.

1. Clone the project, replacing ``VQAMD`` with the name of the directory you are creating:

        $ git clone https://github.com/sominwadhwa/vqa_floyd.git VQAMD
        $ cd VQAMD

2. Make sure you have ``python 3.5.x`` running on your local system. If you do, skip this step. In case you don't, head
head [here](https://www.python.org/downloads/).

3. ``virtualenv`` is a tool used for creating isolated 'virtual' python environments. It is advisable to create one here as well (to avoid installing the pre-requisites into the system-root). Do the following within the project directory:

        $ [sudo] pip install virtualenv
        $ virtualenv --system-site-packages VQAMD
        $ source VQAMD/bin/activate

To deactivate later, once you're done with the project, just type ``deactivate``.

4. Install the pre-requisites from ``requirements.txt`` & run ``tests/init.py`` to check if all the required packages were correctly installed:

        $ pip install -r requirements.txt
        $ bash run_me_first_on_floyd.sh

## Contributing to VQA

I welcome contributions to this little project. If you have any new ideas or approaches that you'd like to incorporate here, feel free to open up an issue.

Please refer to each project's style guidelines and guidelines for submitting patches and additions. In general, we follow the "fork-and-pull" Git workflow.

 1. **Fork** the repo VQAMD on GitHub
 2. **Clone** the project to your own machine
 3. **Commit** changes to your own branch
 4. **Push** your work back up to your fork
 5. Submit a **Pull request** so that we can review your changes

NOTE: Be sure to merge the latest from "upstream" before making a pull request!

### Issues

Feel free to submit issues and enhancement requests.