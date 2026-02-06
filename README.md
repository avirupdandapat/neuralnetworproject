# neuralnetworproject

## Summary
- Mixed ML playground with local-LLM query scripts, classical ML notebooks, RL experiments, and computer-vision work.
- Core themes: EfficientNet-based image classification (bone fracture data), churn prediction, NLP on app reviews, Gymnasium/RL, and causal inference.
- Includes course/assignment notebooks (UC_AIML), Karpathy-style micrograd code, and lecture notebooks from "nn-zero-to-hero".
- Large datasets/models are stored alongside code; see notebook folders for context.


## How to run
- Local LLM scripts
  - `query_llm.py`: start your local Docker Model Runner on `http://localhost:12434`, then run `python query_llm.py`.
  - `query_llm_langchain.py`: install deps with `pip install -r requirements-llm.txt`, then run `python query_llm_langchain.py`.
- Notebooks: open with Jupyter/VS Code and run cells in order. Some notebooks expect local datasets under `neuralnetworproject/`.
- Suggested notebook entrypoints
  - Bone fracture image classification: `neuralnetworproject/EfficientNet.ipynb` or `neuralnetworproject/EfficientNet-Main.ipynb`.
  - Bone image experiments: `neuralnetworproject/boneImage.ipynb`.
  - Churn prediction: `neuralnetworproject/NN-ChurnData.ipynb`.
  - NLP on app reviews: `neuralnetworproject/NLP/mobileappassessment.ipynb`.
  - RL/Gymnasium: `neuralnetworproject/Gymnasium.ipynb` and `neuralnetworproject/RL/RL_Chess.ipynb`.
  - Causal inference: `neuralnetworproject/DoWhy.ipynb`.
  - PyTorch basics: `neuralnetworproject/nn-torch.ipynb`.
  - UC_AIML course work: start with `neuralnetworproject/UC_AIML/Assignment1_2.ipynb` and proceed by assignment number.

## Assets (datasets/models)
- Bone fracture detection images/labels: `neuralnetworproject/Human Bone Fractures Multi-modal Image Dataset (HBFMID)/` and `neuralnetworproject/Human Bone Fractures Multi-modal Image Dataset (HBFMID) main/`.
- Pretrained weights: `neuralnetworproject/efficientnet_b4_rwightman-23ab8bcd.pth`, `neuralnetworproject/best_model.pth`.
- Churn data: `neuralnetworproject/ChurnModeling.csv`.
- NLP reviews data: `neuralnetworproject/inputdata/multilingual_mobile_app_reviews_2025.csv`.
- UC_AIML datasets (CSV/HTML/PNG): `neuralnetworproject/UC_AIML/`.

## Details (all code files)

### Root
- `query_llm.py` — Minimal Python script that calls a locally hosted SmolLM2 model runner via an OpenAI-compatible HTTP endpoint.
- `query_llm_langchain.py` — Same local SmolLM2 query flow but through LangChain `ChatOpenAI` with a custom `base_url`.

### `neuralnetworproject/`
- `EfficientNet.ipynb` — EfficientNet workflow notebook (installation, data loading, training/metrics) for bone-fracture image classification.
- `EfficientNet-Main.ipynb` — Alternate or main run of the EfficientNet workflow.
- `EfficientNet.py` — Notebook exported as `.py` (contains notebook JSON) for the EfficientNet/bone-fracture pipeline.
- `boneImage.ipynb` — Bone image processing/classification notebook.
- `NN-ChurnData.ipynb` — Churn prediction notebook (tabular ML).
- `nn-torch.ipynb` — PyTorch fundamentals/experiments notebook.
- `Gymnasium.ipynb` — Gymnasium-based RL experiments.
- `RL/RL_Chess.ipynb` — Reinforcement-learning experiments in a chess setting.
- `DoWhy.ipynb` — Causal inference experiments using DoWhy-style workflows.
- `NLP/mobileappassessment.ipynb` — NLP notebook analyzing mobile app reviews/assessments.

### `neuralnetworproject/UC_AIML/`
- `Assignment1_2.ipynb` — UC_AIML assignment notebook (foundations).
- `Assignment1_3.ipynb` — UC_AIML assignment notebook (foundations).
- `Assignment2_1.ipynb` — UC_AIML assignment notebook (ML/data prep).
- `Assignment2_2.ipynb` — UC_AIML assignment notebook (ML/data prep).
- `Assignment2_3.ipynb` — UC_AIML assignment notebook (ML/data prep).
- `Assignment2_4.ipynb` — UC_AIML assignment notebook (ML/data prep).
- `Assignment3_1.ipynb` — UC_AIML assignment notebook (modeling).
- `Assignment3_2.ipynb` — UC_AIML assignment notebook (modeling).
- `Assignment3_3.ipynb` — UC_AIML assignment notebook (modeling).
- `Assignment3_4.ipynb` — UC_AIML assignment notebook (modeling).
- `Assignment4_1.ipynb` — UC_AIML assignment notebook (advanced topics).
- `Assignment4_2.ipynb` — UC_AIML assignment notebook (advanced topics).
- `Assignment4_3.ipynb` — UC_AIML assignment notebook (advanced topics).
- `DataVisualization.ipynb` — Data visualization exercises.
- `datavisualization_advance.ipynb` — Advanced data visualization exercises.
- `Dummy.ipynb` — Scratch/placeholder notebook.
- `Jupyter notebook 1.1 Applying linear algebra in Python.ipynb` — Linear algebra foundations in Python.
- `Jupyter notebook 1.2 Applying calculus in Python.ipynb` — Calculus foundations in Python.
- `Jupyter notebook 1.3 Applying probability and statistics in Python.ipynb` — Probability/statistics foundations in Python.
- `MedicalDataPrediction.ipynb` — Medical data prediction exercise.
- `Self_Study_Colab_Activity_2_1.ipynb` — Self-study activity notebook.
- `Self_Study_Colab_Activity_2_1-2.ipynb` — Self-study activity notebook (variant).
- `Self_study_colab_activity_2_2.ipynb` — Self-study activity notebook.
- `Self_study_colab_activity_2_2-2.ipynb` — Self-study activity notebook (variant).
- `Self_study_colab_activity_2_4.ipynb` — Self-study activity notebook.
- `Self_study_try_it_activity_1_1_Machine_learning_foundations.ipynb` — ML foundations try-it activity.
- `Self_study_try_it_activity_1_1_Machine_learning_foundations-2.ipynb` — ML foundations try-it activity (variant).
- `try_it_2-1.ipynb` — Try-it activity notebook.
- `try_it_2-2.ipynb` — Try-it activity notebook.
- `Vector.ipynb` — Vector/matrix operations notebook.

### `neuralnetworproject/micrograd-master/`
- `micrograd/engine.py` — Core scalar autograd engine (Value class and backward pass).
- `micrograd/nn.py` — Simple neural-net building blocks on top of the engine.
- `micrograd/__init__.py` — Package exports.
- `setup.py` — Package metadata/build config.
- `demo.ipynb` — Micrograd demo notebook.
- `trace_graph.ipynb` — Notebook for tracing/visualizing computation graphs.
- `test/test_engine.py` — Unit tests for the engine.

### `neuralnetworproject/nn-zero-to-hero-master/lectures/`
- `micrograd/micrograd_lecture_first_half_roughly.ipynb` — Micrograd lecture notebook (part 1).
- `micrograd/micrograd_lecture_second_half_roughly.ipynb` — Micrograd lecture notebook (part 2).
- `makemore/makemore_part1_bigrams.ipynb` — Makemore lecture notebook (bigrams).
- `makemore/makemore_part2_mlp.ipynb` — Makemore lecture notebook (MLP).
- `makemore/makemore_part3_bn.ipynb` — Makemore lecture notebook (batch norm).
- `makemore/makemore_part4_backprop.ipynb` — Makemore lecture notebook (backprop).
- `makemore/makemore_part5_cnn1.ipynb` — Makemore lecture notebook (CNN).
- `makemore/WebScraping.ipynb` — Web scraping notebook in the lecture folder.
