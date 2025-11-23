# ML Engineer Portfolio Projects

This document provides detailed specifications for 4 portfolio projects that will demonstrate your ML engineering skills for interviews.

---

## Project Selection Strategy

These projects are chosen to:
1. **Cover different ML domains**: Classical ML, Deep Learning (CV + NLP), Modern LLMs
2. **Show end-to-end skills**: Data processing, training, evaluation, deployment
3. **Demonstrate engineering**: Not just notebooks, but production-quality code
4. **Relate to your background**: Leverage your production ML experience where possible
5. **Be interview-ready**: Can discuss architecture, trade-offs, improvements

**Modern MLE Focus (40% Modeling / 60% Engineering)**:
Modern senior MLE roles require production-ready systems, not just high-accuracy models. These projects emphasize:
- **MLOps**: CI/CD, model versioning, artifact management
- **Serving**: APIs, latency optimization, cost-performance trade-offs
- **Infrastructure**: Docker, cloud deployment, monitoring
- **Production patterns**: Feature stores, drift detection, evaluation frameworks

This 40/60 split reflects real-world senior MLE work: building robust, scalable ML systems.

**Timeline**: Complete all 4 projects over 10 weeks (Weeks 1-10 of study plan)

---

## Project 1: Image Classification with Transfer Learning

**Duration**: 2 weeks (Weeks 1-2)
**Difficulty**: Easy-Medium
**Domain**: Computer Vision, Deep Learning

### Objective
Build an image classification system using transfer learning, demonstrating understanding of CNNs, fine-tuning, and model deployment.

### Problem Statement
Classify images into categories (e.g., different types of food, animals, or objects). Use a pre-trained model and fine-tune for your specific dataset.

### Dataset Options
1. **Food-101** - 101 food categories (easy, clean data)
2. **Stanford Dogs** - 120 dog breeds (medium difficulty)
3. **Fashion MNIST** - Clothing items (very easy, good starting point)
4. **Caltech-101** - 101 object categories
5. **Custom dataset** - Scrape your own (more impressive)

**Recommendation**: Start with Food-101 or Fashion MNIST

### Technical Requirements

**1. Data Pipeline**:
- [ ] Load and explore dataset (understand distribution)
- [ ] Split into train/val/test (80/10/10 or 70/15/15)
- [ ] Implement data augmentation (rotation, flip, zoom, color jitter)
- [ ] Create efficient data loaders (PyTorch DataLoader or tf.data)
- [ ] Handle class imbalance (if applicable)

**2. Model Development**:
- [ ] Baseline: Simple CNN from scratch
- [ ] Transfer learning: Use pre-trained model (ResNet50, EfficientNet, or VGG)
- [ ] Fine-tuning strategy:
  - Option A: Freeze early layers, train only top layers
  - Option B: Unfreeze and fine-tune with low learning rate
- [ ] Compare performance of baseline vs. transfer learning

**3. Training**:
- [ ] Implement training loop with validation
- [ ] Use appropriate loss function (CrossEntropyLoss)
- [ ] Optimizer: Adam or SGD with momentum
- [ ] Learning rate scheduling (ReduceLROnPlateau or Cosine)
- [ ] Early stopping
- [ ] Save best model checkpoint

**4. Evaluation**:
- [ ] Metrics: Accuracy, precision, recall, F1 per class
- [ ] Confusion matrix visualization
- [ ] Visualize misclassifications (understand failures)
- [ ] Class activation maps (Grad-CAM) for interpretability

**5. Engineering Quality**:
- [ ] Clean, modular code (separate files for data, model, train, eval)
- [ ] Config file for hyperparameters (YAML or JSON)
- [ ] Logging (track training progress)
- [ ] Requirements.txt with dependencies
- [ ] Command-line interface (argparse) for training/inference

**6. MLOps & Deployment** (Mandatory):
- [ ] **Docker**: Containerize application with Dockerfile
  - Multi-stage build (training vs. inference images)
  - Include model serving endpoint in container
- [ ] **CI/CD**: GitHub Actions workflow
  - Linting (flake8, black)
  - Unit tests (pytest)
  - Trigger on push to main
- [ ] **Model Serving**: FastAPI endpoint
  - POST /predict with image upload
  - Return predictions with confidence scores
  - Add input validation and error handling
- [ ] **Model Versioning**: DVC for artifact management
  - Track model checkpoints
  - Version datasets
  - Remote storage (S3, GCS, or DVC remote)
- [ ] **Cloud Deployment**: Deploy to cloud platform
  - AWS (EC2, Lambda, or SageMaker)
  - GCP (Cloud Run or Vertex AI)
  - Hugging Face Spaces (easiest)
  - Include deployment instructions in README

### Tech Stack
- **Framework**: PyTorch (preferred) or TensorFlow/Keras
- **Pre-trained models**: torchvision.models or tf.keras.applications
- **Visualization**: matplotlib, seaborn
- **Experiment tracking**: Weights & Biases or TensorBoard
- **MLOps**: Docker, GitHub Actions, DVC
- **Serving**: FastAPI, uvicorn
- **Cloud**: AWS/GCP/Hugging Face Spaces

### Deliverables

**Code**:
```
image-classifier/
├── README.md
├── requirements.txt
├── config.yaml
├── Dockerfile           # Docker container
├── .dockerignore
├── .github/
│   └── workflows/
│       └── ci.yml       # CI/CD pipeline
├── data/
│   └── download_data.py
├── src/
│   ├── dataset.py       # Data loading and augmentation
│   ├── model.py         # Model architectures
│   ├── train.py         # Training loop
│   ├── evaluate.py      # Evaluation metrics
│   ├── api.py           # FastAPI endpoint
│   └── utils.py         # Helper functions
├── tests/
│   ├── test_dataset.py  # Unit tests
│   ├── test_model.py
│   └── test_api.py
├── notebooks/
│   └── exploration.ipynb  # Data exploration
├── scripts/
│   ├── train.sh         # Training script
│   ├── evaluate.sh      # Evaluation script
│   └── deploy.sh        # Deployment script
├── outputs/
│   ├── checkpoints/     # Saved models (tracked by DVC)
│   └── results/         # Visualizations
└── .dvc/                # DVC config
    └── config
```

**README.md should include**:
- Problem description and motivation
- Dataset details
- Model architecture and approach
- Results (metrics, visualizations)
- How to run (setup, training, inference)
- Future improvements

### What Interviewers Look For
- Understanding of transfer learning (why it works)
- Data augmentation choices (domain-appropriate)
- Model selection rationale
- Handling overfitting (regularization, augmentation, early stopping)
- Evaluation beyond just accuracy
- Clean, production-ready code

### Discussion Points for Interviews
- "Why did you choose ResNet over other architectures?"
- "How did you decide which layers to fine-tune?"
- "What data augmentation techniques did you use and why?"
- "Walk me through your CI/CD pipeline. What tests do you run?"
- "How does your Docker multi-stage build work? Why separate training/inference?"
- "How do you version your models and datasets with DVC?"
- "What's your FastAPI endpoint design? How do you handle errors?"
- "How would you handle new classes in production?"
- "What's the model's inference time? How could you optimize it?"
- "How do you monitor model performance in production?"

---

## Project 2: Text Classification with Multiple Approaches

**Duration**: 2 weeks (Weeks 2-3)
**Difficulty**: Medium
**Domain**: NLP, Classical ML, Deep Learning

### Objective
Build a text classification system comparing classical ML, deep learning, and transformer approaches. Show understanding of NLP fundamentals and evolution.

### Problem Statement
Classify text into categories. Compare performance of different approaches.

### Dataset Options
1. **IMDB Reviews** - Sentiment analysis (binary, 25k samples)
2. **AG News** - News article classification (4 classes)
3. **20 Newsgroups** - Topic classification (20 classes)
4. **Yelp Reviews** - Sentiment (5-star rating)
5. **Custom**: Scrape tweets or Reddit posts

**Recommendation**: IMDB (easy to start) or AG News (multi-class)

### Technical Requirements

**1. Data Pipeline**:
- [ ] Load and explore dataset (text length distribution, class balance)
- [ ] Text preprocessing:
  - Lowercasing, removing punctuation
  - Stopword removal (optional, compare with/without)
  - Stemming or Lemmatization
- [ ] Train/val/test split (stratified)

**2. Approach 1: Classical ML**:
- [ ] Feature extraction:
  - Bag of Words (CountVectorizer)
  - TF-IDF (TfidfVectorizer)
  - N-grams (unigrams, bigrams)
- [ ] Models:
  - Logistic Regression (baseline)
  - Naive Bayes
  - Random Forest or XGBoost
- [ ] Hyperparameter tuning (GridSearchCV or RandomizedSearchCV)

**3. Approach 2: Deep Learning (RNN/LSTM)**:
- [ ] Tokenization (word-level or subword)
- [ ] Word embeddings:
  - Random initialization (train from scratch)
  - Pre-trained: GloVe or Word2Vec
- [ ] Model architectures:
  - Simple RNN
  - LSTM or GRU
  - Bidirectional LSTM
- [ ] Experiment with embedding dimensions, hidden sizes

**4. Approach 3: Transformers**:
- [ ] Fine-tune pre-trained model:
  - BERT (bert-base-uncased)
  - DistilBERT (faster, smaller)
  - RoBERTa (often better performance)
- [ ] Use Hugging Face Transformers library
- [ ] Implement with Trainer API or custom training loop

**5. Comparison & Analysis**:
- [ ] Compare all approaches on same test set
- [ ] Metrics: Accuracy, F1, precision, recall
- [ ] Training time and inference time
- [ ] Model size
- [ ] Error analysis: Where does each model fail?
- [ ] Create comparison table and visualizations

**6. Production Serving & Optimization** (Mandatory):
- [ ] **Model Quantization**: Optimize transformer model
  - INT8 quantization using PyTorch or ONNX Runtime
  - Compare FP32 vs INT8: accuracy drop vs. speedup
  - Target: <5% accuracy drop, 2-4× speedup
- [ ] **Serving API**: FastAPI endpoint
  - POST /classify endpoint with text input
  - Return prediction + confidence + latency
  - Batch inference support (process multiple texts)
- [ ] **Load Testing**: Measure throughput vs. latency
  - Use Locust or Apache Bench
  - Test scenarios: 1, 10, 50, 100 concurrent users
  - Generate graphs: QPS vs. latency, batch size impact
  - Compare: Classical ML (low latency) vs. Transformer (high accuracy)
- [ ] **Dynamic Batching** (Optional):
  - Implement batch collection with timeout
  - Measure: collection time, batch size, average latency
  - Apply Day 25 learnings: QPS = Batch Size / Inference Time
- [ ] **Docker & Deployment**:
  - Containerize serving endpoint
  - Deploy to cloud (AWS Lambda, GCP Cloud Run, or HF Spaces)
  - Include deployment instructions

### Tech Stack
- **Classical ML**: scikit-learn
- **Deep Learning**: PyTorch or TensorFlow
- **Transformers**: Hugging Face Transformers
- **Tokenization**: NLTK, spaCy, or Hugging Face tokenizers
- **Optimization**: ONNX Runtime, PyTorch quantization, TensorRT (optional)
- **Serving**: FastAPI, uvicorn
- **Load Testing**: Locust, Apache Bench
- **Deployment**: Docker, AWS/GCP/HF Spaces

### Deliverables

**Code Structure**:
```
text-classifier/
├── README.md
├── requirements.txt
├── Dockerfile
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── download_data.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_classical_ml.ipynb
│   ├── 03_deep_learning.ipynb
│   ├── 04_transformers.ipynb
│   └── 05_quantization.ipynb
├── src/
│   ├── preprocessing.py
│   ├── classical_models.py
│   ├── dl_models.py
│   ├── transformer_models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── quantize.py         # Quantization scripts
│   ├── api.py              # FastAPI serving endpoint
│   └── serving.py          # Inference logic
├── tests/
│   ├── test_models.py
│   └── test_api.py
├── load_tests/
│   ├── locustfile.py       # Load testing scenarios
│   └── run_load_test.sh
├── outputs/
│   ├── results/
│   │   ├── comparison_table.csv
│   │   └── load_test_results/  # QPS vs latency graphs
│   └── visualizations/
└── models/
    ├── best_model.pth
    ├── quantized_model.onnx    # INT8 quantized model
    └── model_comparison.json   # FP32 vs INT8 metrics
```

**README should include**:
- Comparison table of all approaches
- Key insights (which worked best and why)
- Trade-offs (accuracy vs. speed vs. complexity)

### What Interviewers Look For
- Understanding of text representation (BoW, TF-IDF, embeddings)
- Knowledge of model evolution (classical → RNN → Transformer)
- Trade-off analysis (when to use what)
- Can explain attention mechanism
- Understanding of overfitting in NLP (small vocab vs. large vocab)

### Discussion Points
- "Why do transformers outperform LSTMs?"
- "When would you use classical ML over deep learning?"
- "How do you handle out-of-vocabulary words?"
- "What's the difference between BERT and GPT?"
- "How would you handle class imbalance?"
- "Walk me through your quantization approach. What accuracy-speed trade-off did you achieve?"
- "Show me your load testing results. How does QPS relate to latency?"
- "How does batch size affect throughput and latency in your serving endpoint?"
- "When would you choose classical ML over transformers in production?"
- "How would you deploy this with <100ms p99 latency at 10K QPS?"

---

## Project 3: Structured Data Prediction with Feature Engineering

**Duration**: 2 weeks (Weeks 3-4)
**Difficulty**: Medium
**Domain**: Classical ML, Feature Engineering, Model Deployment

### Objective
Build a complete ML pipeline for tabular data, emphasizing feature engineering, model selection, and production-ready code.

### Problem Statement
Predict outcome on structured/tabular data. Show strong feature engineering and classical ML skills.

### Dataset Options
1. **House Prices** (Kaggle) - Regression, lots of features
2. **Titanic** - Binary classification (easy, good starting point)
3. **Credit Card Fraud Detection** - Imbalanced classification
4. **Customer Churn** - Binary classification, business-focused
5. **Bike Sharing Demand** - Regression with time series aspect

**Recommendation**: House Prices (regression) or Credit Card Fraud (imbalanced)

### Technical Requirements

**1. Data Exploration & Cleaning**:
- [ ] Exploratory Data Analysis (EDA)
  - Distributions, correlations
  - Missing value analysis
  - Outlier detection
- [ ] Handle missing values (imputation strategies)
- [ ] Handle outliers (remove, cap, or keep)
- [ ] Data types (correct categorical vs. numerical)

**2. Feature Engineering** (Most Important):
- [ ] **Numerical features**:
  - Scaling (StandardScaler, MinMaxScaler, RobustScaler)
  - Log transformation (for skewed distributions)
  - Polynomial features
  - Binning (discretization)
- [ ] **Categorical features**:
  - One-hot encoding
  - Label encoding (for ordinal)
  - Target encoding (with cross-validation to avoid leakage)
  - Frequency encoding
- [ ] **DateTime features** (if applicable):
  - Extract: year, month, day, day_of_week, hour
  - Cyclical encoding (sin/cos for hour, month)
  - Time since event
- [ ] **Interaction features**:
  - Feature crosses (e.g., location × day_of_week)
  - Domain-specific (e.g., total_rooms / total_people)
- [ ] **Aggregation features**:
  - Group statistics (mean, std by category)
- [ ] Feature selection:
  - Correlation analysis
  - Feature importance from tree models
  - Recursive feature elimination

**3. Model Development**:
- [ ] Baseline: Simple model (linear regression, logistic regression)
- [ ] Try multiple algorithms:
  - Linear models (Ridge, Lasso, ElasticNet)
  - Tree-based (Decision Tree, Random Forest)
  - Gradient Boosting (XGBoost, LightGBM, CatBoost)
- [ ] Hyperparameter tuning (Optuna, GridSearchCV, or RandomizedSearchCV)
- [ ] Ensemble methods:
  - Voting/Averaging
  - Stacking

**4. Model Evaluation**:
- [ ] K-fold cross-validation
- [ ] Appropriate metrics:
  - Regression: RMSE, MAE, R²
  - Classification: Accuracy, Precision, Recall, F1, AUC-ROC
- [ ] Learning curves (diagnose overfitting/underfitting)
- [ ] Residual analysis (for regression)
- [ ] Feature importance visualization

**5. Production-Ready Code**:
- [ ] Sklearn Pipeline (preprocessing + model)
- [ ] Save model (joblib or pickle)
- [ ] Inference script (load model, predict on new data)
- [ ] Unit tests for key functions
- [ ] API endpoint (Flask or FastAPI)

**6. ML Infrastructure & Monitoring** (Mandatory):
- [ ] **Orchestration with Airflow**:
  - Create DAG for training pipeline
  - Tasks: data extraction → preprocessing → feature engineering → training → evaluation
  - Schedule: daily or weekly retraining
  - Implement idempotency (Day 18 learnings)
  - Handle failures with retries and alerts
- [ ] **Feature Store Pattern**:
  - Separate feature computation from model training
  - Store features: online (Redis/low-latency) + offline (S3/batch)
  - Implement point-in-time correctness (Day 17 learnings)
  - Ensure train/serve consistency
- [ ] **Drift Detection**:
  - Use EvidentlyAI or Alibi Detect
  - Monitor: data drift (input distributions) + concept drift (target distribution)
  - Generate drift reports: compare training vs. production data
  - Set up alerts for drift thresholds
  - Example: If house price median shifts >20%, trigger alert
- [ ] **Model Monitoring**:
  - Track performance metrics over time
  - Log predictions for analysis
  - A/B testing setup (shadow deployment)
- [ ] **Docker & Deployment**:
  - Containerize training pipeline and serving endpoint
  - Deploy to cloud (AWS SageMaker, GCP Vertex AI, or custom)
  - Include Airflow deployment instructions

### Tech Stack
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **ML**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Hyperparameter tuning**: Optuna or scikit-learn
- **Orchestration**: Apache Airflow
- **Feature Store**: Redis (online), S3/Parquet (offline)
- **Drift Detection**: EvidentlyAI or Alibi Detect
- **API**: FastAPI
- **Deployment**: Docker, AWS/GCP

### Deliverables

**Code Structure**:
```
structured-ml-pipeline/
├── README.md
├── requirements.txt
├── config.yaml
├── Dockerfile
├── docker-compose.yml     # Airflow + Redis setup
├── data/
│   ├── raw/
│   ├── processed/
│   └── monitoring/        # Drift reports
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_drift_analysis.ipynb
├── dags/
│   ├── training_pipeline.py      # Airflow DAG
│   └── feature_refresh_dag.py    # Feature store updates
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── feature_store.py          # Feature store logic
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── drift_detection.py        # EvidentlyAI integration
│   └── monitoring.py             # Performance tracking
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_api.py
├── models/
│   ├── best_model.pkl
│   └── model_metadata.json       # Version, metrics, training date
├── api/
│   └── app.py                    # FastAPI endpoint
└── monitoring/
    ├── drift_reports/            # Generated drift reports
    └── performance_logs/         # Model performance over time
```

**README highlights**:
- Feature engineering decisions (most important section!)
- Model comparison table
- Performance on test set
- How to use for inference

### What Interviewers Look For
- **Strong feature engineering** (this is the most important skill)
- Understanding of different encodings for categorical features
- Knowledge of when to use which model
- Handling data issues (missing values, outliers, imbalance)
- Understanding of regularization
- Production considerations (pipelines, model saving/loading)

### Discussion Points
- "Walk me through your feature engineering process"
- "Why did you choose XGBoost over Random Forest?"
- "How did you handle missing values?"
- "Explain your Airflow DAG. How do you ensure idempotency?"
- "How does your feature store work? How do you ensure train/serve consistency?"
- "Walk me through your drift detection approach. What triggers a model retrain?"
- "How would you handle new categories at inference time?"
- "How would you update this model in production without downtime?"
- "What's the risk of data leakage in your pipeline? How did you prevent it?"
- "Show me a drift report. What would you do if you detected significant drift?"

---

## Project 4: LLM-Based Application (RAG or Fine-tuning)

**Duration**: 2 weeks (Weeks 5-6)
**Difficulty**: Medium-Hard
**Domain**: Modern AI, LLMs, Production Systems

**NOTE**: This project creates a portfolio piece showcasing LLM/RAG skills. Draw from production learnings about prompt reliability, cost optimization, latency considerations, and RAG architecture design.

### Objective
Build a practical LLM application demonstrating understanding of modern AI techniques. Choose between RAG or fine-tuning based on interest.

### Option A: RAG System (Recommended)

**Problem Statement**: Build a question-answering system over a custom knowledge base using Retrieval-Augmented Generation.

**Example Use Cases**:
1. **Documentation Q&A**: Answer questions about a software library
2. **Personal Knowledge Base**: Q&A over your notes/articles
3. **Customer Support Bot**: Answer FAQs from company docs
4. **Research Assistant**: Q&A over academic papers

**Technical Requirements**:

**1. Data Collection**:
- [ ] Collect documents (scrape docs, PDFs, or use dataset)
- [ ] Process and clean text
- [ ] Chunk documents (500-1000 tokens, with overlap)
- [ ] Extract metadata (source, date, section)

**2. Embedding & Indexing**:
- [ ] Choose embedding model:
  - Sentence-BERT (all-MiniLM-L6-v2)
  - OpenAI embeddings
  - Instructor embeddings
- [ ] Generate embeddings for all chunks
- [ ] Store in vector database:
  - Pinecone (cloud, easy)
  - Weaviate (cloud/local)
  - FAISS (local, fast)
  - Chroma (local, simple)

**3. Retrieval**:
- [ ] Implement semantic search
- [ ] Retrieve top-K relevant chunks (K=3-5)
- [ ] Experiment with retrieval strategies:
  - Dense retrieval (embeddings only)
  - Hybrid (dense + BM25 sparse)
  - Reranking (retrieve 20, rerank to top 5)

**4. Generation**:
- [ ] Choose LLM:
  - OpenAI GPT-4 or GPT-3.5-turbo
  - Anthropic Claude
  - Open-source: Llama 2, Mistral (via Ollama or HF)
- [ ] Prompt engineering:
  - System prompt with instructions
  - Include retrieved context
  - Ask for citations
- [ ] Handle context window limits

**5. Automated Evaluation with Frameworks** (Mandatory):
- [ ] **Create Test Question Set**:
  - 50-100 questions with ground truth answers
  - Include: simple factual, complex reasoning, multi-hop
  - Negative samples (questions with no answer in corpus)
- [ ] **Retrieval Metrics** (Manual):
  - Recall@K (K=3,5,10): Are correct docs in top K?
  - Precision@K: How many retrieved docs are relevant?
  - MRR (Mean Reciprocal Rank): Position of first correct doc
  - NDCG: Normalized Discounted Cumulative Gain
- [ ] **RAG Evaluation with Ragas** (Automated LLM-as-judge):
  - **Context Precision**: Are retrieved contexts relevant to question?
  - **Context Recall**: Does retrieved context contain answer?
  - **Faithfulness**: Is answer grounded in retrieved context (no hallucination)?
  - **Answer Relevance**: Does answer address the question?
  - **Answer Correctness**: Semantic similarity with ground truth
- [ ] **Alternative: DeepEval Framework**:
  - Similar metrics to Ragas
  - G-Eval: Custom evaluation criteria
  - Hallucination detection
- [ ] **Comparison Studies**:
  - Compare embedding models (sentence-transformers vs. OpenAI)
  - Compare retrieval strategies (dense vs. hybrid vs. reranking)
  - Ablation: retrieval-only vs. full RAG
- [ ] **Error Analysis**:
  - Categorize failures: retrieval miss, poor generation, hallucination
  - Analyze: which question types fail most?
  - Visualize: score distributions, failure modes
- [ ] **Cost-Performance Trade-off**:
  - Track: API calls, tokens used, latency per query
  - Calculate: cost per 1K queries
  - Compare: GPT-4 (expensive, accurate) vs. GPT-3.5 (cheap, fast)

**6. User Interface**:
- [ ] CLI interface (minimum)
- [ ] Web UI (Streamlit, Gradio) - recommended
- [ ] Show sources/citations
- [ ] Conversation history (optional)

**7. Deployment & Infrastructure** (Mandatory):
- [ ] **Docker**: Containerize RAG application
  - Include vector DB (FAISS/Chroma) or connect to cloud DB
  - Environment variables for API keys
  - Multi-stage build if using local models
- [ ] **Cloud Deployment**:
  - Streamlit: Deploy to Streamlit Cloud or HF Spaces (easiest)
  - API: Deploy FastAPI to AWS Lambda, GCP Cloud Run, or Railway
  - Vector DB: Use cloud service (Pinecone, Weaviate Cloud) for production
- [ ] **Monitoring**:
  - Log queries, retrieval results, and responses
  - Track latency and costs
  - Optional: Set up feedback mechanism for users

**Tech Stack**:
- **Embeddings**: sentence-transformers, OpenAI API
- **Vector DB**: FAISS, Chroma, Pinecone, or Weaviate
- **LLM**: OpenAI API, Anthropic API, or local (Ollama)
- **Orchestration**: LangChain or LlamaIndex (optional, can do from scratch)
- **Evaluation**: Ragas, DeepEval, BEIR (retrieval benchmark)
- **UI**: Streamlit or Gradio
- **Deployment**: Docker, Streamlit Cloud, HF Spaces, AWS/GCP

**Code Structure**:
```
rag-qa-system/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml    # Optional: multi-service setup
├── .env.example          # API keys template
├── .github/
│   └── workflows/
│       └── ci.yml        # CI for testing
├── data/
│   ├── raw/              # Original documents
│   ├── processed/        # Chunked documents
│   └── eval/             # Test question sets
├── src/
│   ├── data_loader.py    # Load and chunk documents
│   ├── embeddings.py     # Generate embeddings
│   ├── vector_store.py   # Vector DB operations
│   ├── retriever.py      # Retrieve relevant docs
│   ├── generator.py      # LLM generation
│   ├── rag_pipeline.py   # End-to-end pipeline
│   └── api.py            # FastAPI endpoint (optional)
├── evaluation/
│   ├── test_questions.json
│   ├── evaluate_retrieval.py    # Recall@K, MRR, NDCG
│   ├── evaluate_rag.py          # Ragas/DeepEval integration
│   ├── error_analysis.py        # Categorize failures
│   └── cost_analysis.py         # Track API costs
├── app.py                # Streamlit/Gradio UI
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_embedding_comparison.ipynb
│   ├── 03_retrieval_tuning.ipynb
│   └── 04_evaluation_analysis.ipynb
├── tests/
│   ├── test_pipeline.py
│   └── test_api.py
└── outputs/
    ├── eval_results/     # Evaluation metrics and reports
    └── logs/             # Query logs and monitoring
```

### Option B: Fine-tuning

**Problem Statement**: Fine-tune a pre-trained model for a specific task.

**Example Tasks**:
1. **Text Classification**: Fine-tune BERT for domain-specific classification
2. **Named Entity Recognition**: Custom entity types
3. **Summarization**: Fine-tune T5 or BART
4. **Code Generation**: Fine-tune small model on specific codebase

**Technical Requirements**:
- [ ] Prepare training data (labeled examples)
- [ ] Choose base model (BERT, DistilBERT, T5, GPT-2, Llama 2)
- [ ] Implement fine-tuning:
  - Full fine-tuning (if small model)
  - LoRA or QLoRA (if large model)
- [ ] Evaluation on test set
- [ ] Compare with base model (show improvement)
- [ ] Deploy fine-tuned model

**Tech Stack**: Hugging Face Transformers, PEFT (for LoRA), Weights & Biases

---

### What Interviewers Look For (Project 4)

**For RAG**:
- Understanding of embeddings and similarity search
- Prompt engineering skills
- Chunking strategy rationale
- **Rigorous evaluation methodology** (Ragas, retrieval metrics)
- Ability to measure and improve each component (retrieval, generation)
- Error analysis and failure mode understanding
- Handling context window limits
- Production considerations (cost, latency, monitoring)

**For Fine-tuning**:
- Data preparation for fine-tuning
- Understanding of transfer learning
- Knowledge of efficient fine-tuning (LoRA)
- Evaluation methodology
- Preventing catastrophic forgetting

### Discussion Points
- "How do you handle hallucinations?" (Answer: Faithfulness metric in Ragas)
- "How do you ensure retrieved documents are relevant?" (Context precision, Recall@K)
- "What's your chunking strategy and why?"
- "Walk me through your evaluation methodology. Show me your Ragas scores."
- "What's the difference between context recall and answer relevance?"
- "How did you measure retrieval quality independently of generation?"
- "Show me your error analysis. What are the main failure modes?"
- "How would you scale this to millions of documents?"
- "What's the cost-accuracy trade-off between GPT-4 and GPT-3.5?"
- "How do you measure quality of RAG system?" (Automated eval frameworks!)
- "What's the cost of this system at scale?"
- "How would you update the knowledge base without downtime?"

---

## Project Presentation Tips

For each project, prepare to discuss:

### Architecture
- Draw system diagram
- Explain data flow
- Justify design decisions

### Challenges
- What was hard?
- What didn't work initially?
- How did you debug?

### Results
- Key metrics
- Comparison with baselines
- Error analysis

### Improvements
- What would you do with more time?
- How would you scale?
- What's missing for production?

### Trade-offs
- Accuracy vs. speed
- Complexity vs. maintainability
- Cost vs. performance

---

## GitHub Best Practices

For all projects:

**README Structure**:
1. Title and one-line description
2. Demo (GIF or screenshot)
3. Problem statement
4. Approach/architecture
5. Results
6. Installation and usage
7. Future work

**Code Quality**:
- [ ] Type hints (Python 3.7+)
- [ ] Docstrings
- [ ] Consistent naming
- [ ] No hardcoded paths
- [ ] Config files for hyperparameters
- [ ] requirements.txt with versions

**Git Practices**:
- [ ] Meaningful commit messages
- [ ] .gitignore (don't commit data, models, credentials)
- [ ] Branches for features (optional but professional)

**Additional Enhancements** (Optional but impressive):
- [ ] Advanced monitoring dashboards (Grafana, Prometheus)
- [ ] A/B testing framework for model updates
- [ ] Comprehensive documentation (Sphinx or MkDocs)
- [ ] Technical blog post explaining architecture and learnings
- [ ] Performance profiling and optimization reports
- [ ] Multi-region deployment setup
- [ ] Auto-scaling configurations

---

## Timeline Summary

| Weeks | Project | Domain |
|-------|---------|--------|
| 1-2 | Image Classification | Computer Vision |
| 2-3 | Text Classification | NLP (Classical + DL) |
| 3-4 | Structured Data | Feature Engineering |
| 5-6 | LLM Application | Modern AI (RAG/Fine-tuning) |
| 7-10 | Polish all projects | - |

---

## Additional Project Ideas (If Time)

If you finish early or want a 5th project:

1. **Recommendation System**: Build collaborative filtering or content-based recommender
2. **Time Series Forecasting**: Predict stock prices or weather
3. **Object Detection**: Use YOLO or Faster R-CNN
4. **Speech Recognition**: Fine-tune Wav2Vec2
5. **ML Deployment**: Dockerize model, create REST API, deploy to cloud
6. **AutoML**: Build simple AutoML pipeline
7. **Anomaly Detection**: Isolation Forest or Autoencoder for outlier detection

---

## Resources

**Datasets**:
- Kaggle: https://www.kaggle.com/datasets
- UCI ML Repository: https://archive.ics.uci.edu/ml/
- Hugging Face Datasets: https://huggingface.co/datasets
- Papers with Code: https://paperswithcode.com/datasets

**Inspiration**:
- Kaggle kernels (see what others have done)
- GitHub trending ML projects
- ML competition winning solutions

**Deployment**:
- Streamlit: https://streamlit.io/
- Gradio: https://gradio.app/
- FastAPI: https://fastapi.tiangolo.com/
- Hugging Face Spaces: https://huggingface.co/spaces

---

**Start with Project 1 this week!** Build progressively, and by Week 10 you'll have a strong portfolio that demonstrates end-to-end ML engineering skills. Good luck!
