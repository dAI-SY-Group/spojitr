# Spojit&#344; - Intelligently link development artifacts

The **spojit&#344;** (spojit &#345;&#237;zen&#237;) tool and workflow engine assists in creating trace links between development artifacts stored in issue tracking systems (ITS) and version control systems (VCS).
The current implementation specifically targets the combination of _git_ as VCS and _Atlassian Jira_ as ITS, but the underlying [spojit](https://github.com/michaelrath-work/spojit) library is independent of particular tools.

## Quick Start

You can watch an introduction video on youtube: https://www.youtube.com/watch?v=-zwN6p4Q0jo

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/-zwN6p4Q0jo/0.jpg)](https://www.youtube.com/watch?v=-zwN6p4Q0jo)

Further, we created an [interactive demonstration](demo.asciidoc) within a docker container to get a first impression about **spojit&#344;** and its capabilities.

## Why spojit&#344;

**Spojit&#344;** intelligently _tags_ git commit messages with a jira issue identifier.
This creates a navigable trace link from the jira issue to the commit and vice versa.
Tagging is a commonly applied practice and the Apache Software Foundation even mandates it in its [contribution guidelines](http://apache.org/dev/committers.html#applying-patches).
Once trace links are established, they can be used to  quickly identify code locations where a particular issue is implemented.
On the other hand, on a commit level, the link answers why a certain code change was made by referring to an issue.
However, manually managing, i.e. looking up ids in Jira, and adding the issue ids is tedious and prone to errors.
Developers make (spelling) mistakes when entering the ids or simply forget to do so.
**Spojit&#344;** tries to solve these problems and intelligently automates the tagging process.

## How it works

**Spojit&#344;** uses a [`git hook`](https://git-scm.com/docs/githooks) to extend the usual git workflow.
Whenever a developer performs a commit, **spojit&#344;** analyzes this commit and allows to link it to one of the issues present in the issue tracking system.
Therefore it recommends the three most likely issues ids and asks the user to pick the appropriate one.
The recommendation is build on a machine learning algorithm provided by the [spojit package](https://github.com/michaelrath-work/spojit).
This algorithm is trained on previously performed commits and improved with every new one.

## Installation

Before installing **spojit&#344;** on your local machine, consider to try it within docker (see also respective section in the [demo](demo.asciidoc)).
Use a docker [volume mount](https://docs.docker.com/storage/volumes/) to access your project checkout within the container.

```bash
# build container (see docker_build.sh)
docker build . --tag spojitr

# run container
docker run -it -v <your checkout>:/code/ spojitr
```

### Requirements

**Spojitr** depends on the following 3rd party tools and libraries:

* git
* Java Runtime environment (JRE) >= 8.0
* python >= 3.6
* [spojit](https://github.com/michaelrath-work/spojit)
* [Weka](https://www.cs.waikato.ac.nz/ml/weka/) >= 3.8
* [Weka run helper](https://github.com/michaelrath-work/weka-run-helper)

[`requirements.txt`](requirements.txt) contains the list of required `python packages` that are available via `pip` and can be installed using `pip3 install -r requirements.txt`.

### Local Installation

1. Clone the **spojitr** repository and copy th folder `spojitr_install` folder to your desired installation location
2. Install `spojit`

    ```bash
    git clone https://github.com/michaelrath-work/spojit.git
    cd spojit
    python3 setup.py bdist_wheel
    pip3 install dist/spojit-*.whl
    ```

3. Install `weka run helper`

    ```bash
    git clone https://github.com/michaelrath-work/weka-run-helper.git
    ```

    Copy the `run_weka.py` file to the your `spojitr_install/3rd` folder (see step 1)

4. Copy the `weka.jar` file of the weka installation to your `spojitr_install/3rd` folder (see step 1)

5. Install additional NLTK data files from within a python shell

    ```python
    >>> import nltk
    >>> nltk.download("stopwords")
    >>> nltk.download("word_tokenize")
    >>> nltk.download("punkt")
    ```

6. Run `install.py` within your `spojitr_install` folder (see step 1) to export the `SPOJITRPATH` variable and register the `spojitr` tools in your shell (bash) environment

## Command line usage

In a terminal, type `spojitr -h` to get a list of the available commands and options.

The material presented in [quick start](#Quick-Start), especially the [interactive demonstration](demo.asciidoc) and the video shows how to use **spojit&#344;**.
