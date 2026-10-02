"""
Expanded quality benchmark: 50+ queries across multiple domains.
Measures NDCG@10, MRR, Recall@10 for FP32 vs INT8 with statistical significance.
"""
import argparse
import os
import sys

# Patch cffi
import cffi

_orig = cffi.FFI.cdef
def _patch(self, cs, override=False, **kw):
    return _orig(self, cs, override=True, **kw)
cffi.FFI.cdef = _patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'src'))
from libembedding import Reranker


def compute_dcg(relevances, k=10):
    """Compute DCG@k."""
    import math
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


def compute_ndcg(scores_with_relevance, k=10):
    """Compute NDCG@k."""
    sorted_by_score = sorted(scores_with_relevance, key=lambda x: x[0], reverse=True)
    relevances = [r for _, r in sorted_by_score]
    ideal_relevances = sorted([r for _, r in scores_with_relevance], reverse=True)
    dcg = compute_dcg(relevances, k)
    idcg = compute_dcg(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(scores_with_relevance, relevance_threshold=1):
    """Compute MRR."""
    sorted_by_score = sorted(scores_with_relevance, key=lambda x: x[0], reverse=True)
    for i, (_, rel) in enumerate(sorted_by_score):
        if rel >= relevance_threshold:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall(scores_with_relevance, k=10, relevance_threshold=1):
    """Compute Recall@k."""
    total_relevant = sum(1 for _, r in scores_with_relevance if r >= relevance_threshold)
    if total_relevant == 0:
        return 0.0
    sorted_by_score = sorted(scores_with_relevance, key=lambda x: x[0], reverse=True)
    relevant_in_top_k = sum(1 for _, r in sorted_by_score[:k] if r >= relevance_threshold)
    return relevant_in_top_k / total_relevant


def create_test_corpus():
    """Create 50 test queries across multiple domains."""
    return [
        # === TECHNICAL (10 queries) ===
        ("What is deep learning?", [
            "Deep learning uses neural networks with multiple layers.",
            "Machine learning is a subset of AI.",
            "Pizza is an Italian dish.",
            "Deep learning revolutionized computer vision.",
            "Neural networks are the foundation of deep learning.",
            "Climate change affects global temperatures.",
            "Convolutional neural networks for image recognition.",
            "Python is a programming language.",
            "Recurrent neural networks for sequential data.",
            "The Eiffel Tower is in Paris.",
        ], [2, 1, 0, 2, 2, 0, 1, 0, 1, 0]),

        ("How does gradient descent work?", [
            "Gradient descent optimizes model parameters iteratively.",
            "Backpropagation computes gradients in neural networks.",
            "Pizza is an Italian dish.",
            "Learning rate controls step size in gradient descent.",
            "The human brain has 86 billion neurons.",
            "Stochastic gradient descent uses mini-batches.",
            "Climate change affects temperatures.",
            "Adam optimizer adapts learning rates per parameter.",
            "Shakespeare wrote 37 plays.",
            "Momentum accelerates gradient descent convergence.",
        ], [2, 1, 0, 2, 0, 1, 0, 1, 0, 1]),

        ("What is transfer learning?", [
            "Transfer learning reuses pre-trained models for new tasks.",
            "Fine-tuning adapts a pre-trained model to specific data.",
            "Pizza is an Italian dish.",
            "BERT is pre-trained on large text corpora.",
            "The Eiffel Tower is in Paris.",
            "Domain adaptation transfers knowledge across domains.",
            "Neural networks have multiple layers.",
            "Feature extraction uses pre-trained representations.",
            "Climate change affects temperatures.",
            "Multi-task learning trains on multiple objectives.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("How to prevent overfitting?", [
            "Regularization penalizes model complexity.",
            "Dropout randomly disables neurons during training.",
            "Pizza is an Italian dish.",
            "Early stopping halts training when validation loss increases.",
            "The human brain has 86 billion neurons.",
            "Data augmentation increases training set diversity.",
            "Climate change affects temperatures.",
            "L1/L2 regularization adds penalty terms.",
            "Shakespeare wrote 37 plays.",
            "Cross-validation estimates generalization performance.",
        ], [2, 2, 0, 2, 0, 1, 0, 1, 0, 1]),

        ("What is attention mechanism?", [
            "Attention allows models to focus on relevant input parts.",
            "Self-attention computes relationships between all positions.",
            "Pizza is an Italian dish.",
            "Transformer architecture relies entirely on attention.",
            "The Eiffel Tower is in Paris.",
            "Multi-head attention runs multiple attention functions.",
            "Neural networks have multiple layers.",
            "Attention weights determine importance of each token.",
            "Climate change affects temperatures.",
            "Cross-attention relates two different sequences.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("Explain batch normalization", [
            "Batch normalization normalizes layer inputs per mini-batch.",
            "It reduces internal covariate shift in deep networks.",
            "Pizza is an Italian dish.",
            "Batch norm allows higher learning rates.",
            "The human brain has neurons.",
            "Normalization stabilizes training of deep networks.",
            "Climate change affects temperatures.",
            "Running statistics are used during inference.",
            "Shakespeare wrote plays.",
            "Layer normalization is an alternative to batch norm.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("What is reinforcement learning?", [
            "RL learns through trial and error with rewards.",
            "Q-learning estimates action-value functions.",
            "Pizza is an Italian dish.",
            "Policy gradients optimize action selection directly.",
            "The Eiffel Tower is in Paris.",
            "Deep Q-Networks combine Q-learning with deep learning.",
            "Neural networks have layers.",
            "Actor-critic methods combine value and policy learning.",
            "Climate change affects temperatures.",
            "Exploration vs exploitation is a key RL tradeoff.",
        ], [2, 1, 0, 1, 0, 2, 0, 1, 0, 1]),

        ("How do transformers work?", [
            "Transformers use self-attention to process sequences.",
            "Positional encoding provides sequence order information.",
            "Pizza is an Italian dish.",
            "Multi-head attention captures different representation subspaces.",
            "The human brain has neurons.",
            "Feed-forward networks process each position independently.",
            "Climate change affects temperatures.",
            "Layer normalization stabilizes training.",
            "Shakespeare wrote plays.",
            "Encoder-decoder architecture enables sequence-to-sequence tasks.",
        ], [2, 1, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("What is natural language processing?", [
            "NLP enables computers to understand human language.",
            "Tokenization splits text into words or subwords.",
            "Pizza is an Italian dish.",
            "Named entity recognition identifies proper nouns.",
            "The Eiffel Tower is in Paris.",
            "Sentiment analysis determines text polarity.",
            "Neural networks process text.",
            "Machine translation converts text between languages.",
            "Climate change affects temperatures.",
            "Question answering retrieves answers from text.",
        ], [2, 1, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("Explain computer vision", [
            "CV enables machines to interpret visual information.",
            "Convolutional layers detect local patterns in images.",
            "Pizza is an Italian dish.",
            "Object detection locates and classifies objects.",
            "The human brain has neurons.",
            "Image segmentation partitions images into regions.",
            "Climate change affects temperatures.",
            "Transfer learning from ImageNet improves CV models.",
            "Shakespeare wrote plays.",
            "YOLO is a real-time object detection algorithm.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        # === E-COMMERCE (8 queries) ===
        ("red running shoes", [
            "Nike Air Max red running shoes for men.",
            "Adidas Ultraboost red athletic footwear.",
            "Pizza delivery near me.",
            "New Balance red trail running shoes.",
            "The Eiffel Tower is in Paris.",
            "ASICS Gel-Kayano red stability running shoes.",
            "Climate change affects temperatures.",
            "Pegasus red cushioned running shoes.",
            "Shakespeare wrote plays.",
            "Brooks Ghost red neutral running shoes.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("wireless bluetooth headphones", [
            "Sony WH-1000XM5 wireless noise-canceling headphones.",
            "Bose QuietComfort bluetooth over-ear headphones.",
            "Pizza delivery near me.",
            "Apple AirPods Max wireless headphones.",
            "The Eiffel Tower is in Paris.",
            "Sennheiser Momentum 4 wireless headphones.",
            "Climate change affects temperatures.",
            "JBL Tour One M2 bluetooth headphones.",
            "Shakespeare wrote plays.",
            "Beats Studio Pro wireless headphones.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("stainless steel water bottle", [
            "Hydro Flask 32oz stainless steel insulated bottle.",
            "Yeti Rambler stainless steel water bottle.",
            "Pizza delivery near me.",
            "Klean Kanteen stainless steel reusable bottle.",
            "The Eiffel Tower is in Paris.",
            "S'well stainless steel travel bottle.",
            "Climate change affects temperatures.",
            "Contigo stainless steel autoseal bottle.",
            "Shakespeare wrote plays.",
            "Nalgene wide mouth stainless bottle.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("organic coffee beans", [
            "Blue Bottle organic single-origin coffee beans.",
            "Stumptown organic whole bean coffee.",
            "Pizza delivery near me.",
            "Counter Culture organic fair-trade coffee.",
            "The Eiffel Tower is in Paris.",
            "Intelligentsia organic direct-trade coffee beans.",
            "Climate change affects temperatures.",
            "Counter Culture organic blend coffee.",
            "Shakespeare wrote plays.",
            "Peet's organic dark roast coffee beans.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("yoga mat non-slip", [
            "Manduka PRO non-slip yoga mat.",
            "Liforme eco-friendly non-slip yoga mat.",
            "Pizza delivery near me.",
            "Gaiam premium non-slip yoga mat.",
            "The Eiffel Tower is in Paris.",
            "Jade Harmony natural rubber yoga mat.",
            "Climate change affects temperatures.",
            "Aopoqi extra thick non-slip yoga mat.",
            "Shakespeare wrote plays.",
            "Hugger Mugger alignment yoga mat.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("mechanical keyboard rgb", [
            "Corsair K95 RGB mechanical gaming keyboard.",
            "Razer BlackWidow V4 RGB mechanical keyboard.",
            "Pizza delivery near me.",
            "Logitech G915 low-profile RGB keyboard.",
            "The Eiffel Tower is in Paris.",
            "Keychron K8 wireless mechanical RGB keyboard.",
            "Climate change affects temperatures.",
            "SteelSeries Apex Pro RGB gaming keyboard.",
            "Shakespeare wrote plays.",
            "Ducky One 2 Mini RGB mechanical keyboard.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("portable phone charger", [
            "Anker PowerCore 26800 portable charger.",
            "RAVPower 20000mAh USB-C power bank.",
            "Pizza delivery near me.",
            "Belkin Boost Charge portable battery pack.",
            "The Eiffel Tower is in Paris.",
            "Mophie Powerstation XXL portable charger.",
            "Climate change affects temperatures.",
            "Goal Zero Sherpa 100PD power bank.",
            "Shakespeare wrote plays.",
            "Xiaomi Mi Power Bank 3 Pro.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("leather laptop bag", [
            "Targus Classic leather laptop bag 15.6 inch.",
            "SwissGear leather messenger laptop bag.",
            "Pizza delivery near me.",
            "Kenneth Cole leather laptop briefcase.",
            "The Eiffel Tower is in Paris.",
            "Samsonite leather laptop commuter bag.",
            "Climate change affects temperatures.",
            "AmazonBasics vegan leather laptop sleeve.",
            "Shakespeare wrote plays.",
            "Pad & Quilt leather laptop messenger.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        # === FAQ (8 queries) ===
        ("How to reset my password?", [
            "Click 'Forgot Password' on the login page.",
            "Check your email for password reset link.",
            "Pizza is an Italian dish.",
            "Contact support if you don't receive the email.",
            "The Eiffel Tower is in Paris.",
            "Use your phone number for password recovery.",
            "Climate change affects temperatures.",
            "Create a new strong password with 12+ characters.",
            "Shakespeare wrote plays.",
            "Enable two-factor authentication for security.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("What are your shipping options?", [
            "Standard shipping takes 5-7 business days.",
            "Express shipping available for 2-day delivery.",
            "Pizza is an Italian dish.",
            "Free shipping on orders over $50.",
            "The Eiffel Tower is in Paris.",
            "International shipping available to 50+ countries.",
            "Climate change affects temperatures.",
            "Same-day delivery available in select cities.",
            "Shakespeare wrote plays.",
            "Track your order with the provided tracking number.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("How do I return a product?", [
            "Initiate returns within 30 days of purchase.",
            "Print the prepaid return shipping label.",
            "Pizza is an Italian dish.",
            "Refunds processed within 5-7 business days.",
            "The Eiffel Tower is in Paris.",
            "Items must be in original condition with tags.",
            "Climate change affects temperatures.",
            "Exchange for different size or color available.",
            "Shakespeare wrote plays.",
            "Contact customer service for return authorization.",
        ], [2, 1, 0, 2, 0, 1, 0, 1, 0, 1]),

        ("Is my payment information secure?", [
            "We use 256-bit SSL encryption for all transactions.",
            "PCI DSS Level 1 compliance ensures data security.",
            "Pizza is an Italian dish.",
            "We never store full credit card numbers.",
            "The Eiffel Tower is in Paris.",
            "Tokenization replaces sensitive data with tokens.",
            "Climate change affects temperatures.",
            "Fraud detection monitors suspicious transactions.",
            "Shakespeare wrote plays.",
            "Apple Pay and Google Pay available for secure checkout.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("How to contact customer support?", [
            "Live chat available 24/7 on our website.",
            "Call 1-800-SUPPORT for immediate assistance.",
            "Pizza is an Italian dish.",
            "Email support responds within 24 hours.",
            "The Eiffel Tower is in Paris.",
            "Schedule a callback at your convenience.",
            "Climate change affects temperatures.",
            "Visit our help center for self-service options.",
            "Shakespeare wrote plays.",
            "Social media support via Twitter and Facebook.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("What is your warranty policy?", [
            "All products come with 1-year manufacturer warranty.",
            "Extended warranty available for purchase.",
            "Pizza is an Italian dish.",
            "Covers defects in materials and workmanship.",
            "The Eiffel Tower is in Paris.",
            "Warranty claims processed within 10 business days.",
            "Climate change affects temperatures.",
            "Keep your proof of purchase for warranty claims.",
            "Shakespeare wrote plays.",
            "Accident protection plans available.",
        ], [2, 1, 0, 2, 0, 1, 0, 1, 0, 1]),

        ("Do you offer student discounts?", [
            "10% student discount with valid .edu email.",
            "StudentBeans and UNiDAYS verification accepted.",
            "Pizza is an Italian dish.",
            "Back-to-school specials in August-September.",
            "The Eiffel Tower is in Paris.",
            "Military and educator discounts also available.",
            "Climate change affects temperatures.",
            "Stack discounts with seasonal promotions.",
            "Shakespeare wrote plays.",
            "Sign up for student newsletter for exclusive deals.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("How to track my order?", [
            "Check your email for tracking number after shipment.",
            "Track online using the order tracking portal.",
            "Pizza is an Italian dish.",
            "Real-time GPS tracking available for express orders.",
            "The Eiffel Tower is in Paris.",
            "SMS notifications for delivery updates.",
            "Climate change affects temperatures.",
            "Estimated delivery date shown in order history.",
            "Shakespeare wrote plays.",
            "Contact carrier directly with tracking number.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        # === MEDICAL (6 queries) ===
        ("symptoms of diabetes", [
            "Increased thirst and frequent urination.",
            "Unexplained weight loss and fatigue.",
            "Pizza is an Italian dish.",
            "Blurred vision and slow-healing wounds.",
            "The Eiffel Tower is in Paris.",
            "Tingling or numbness in hands or feet.",
            "Climate change affects temperatures.",
            "Type 1 vs Type 2 diabetes differences.",
            "Shakespeare wrote plays.",
            "HbA1c test for diabetes diagnosis.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("What causes headaches?", [
            "Tension headaches from stress and muscle tightness.",
            "Migraines triggered by hormonal changes.",
            "Pizza is an Italian dish.",
            "Dehydration and caffeine withdrawal.",
            "The Eiffel Tower is in Paris.",
            "Sinus congestion and allergies.",
            "Climate change affects temperatures.",
            "Eye strain from prolonged screen use.",
            "Shakespeare wrote plays.",
            "Cluster headaches are severe one-sided pain.",
        ], [1, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("How to lower blood pressure?", [
            "Reduce sodium intake to under 2300mg daily.",
            "Regular aerobic exercise 30 minutes daily.",
            "Pizza is an Italian dish.",
            "Maintain healthy BMI through diet.",
            "The Eiffel Tower is in Paris.",
            "Limit alcohol consumption.",
            "Climate change affects temperatures.",
            "Manage stress through meditation.",
            "Shakespeare wrote plays.",
            "Take prescribed medications as directed.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("What is pneumonia?", [
            "Pneumonia is lung infection causing inflammation.",
            "Bacterial pneumonia treated with antibiotics.",
            "Pizza is an Italian dish.",
            "Viral pneumonia may require antiviral medication.",
            "The Eiffel Tower is in Paris.",
            "Symptoms include cough fever and difficulty breathing.",
            "Climate change affects temperatures.",
            "Vaccination prevents some types of pneumonia.",
            "Shakespeare wrote plays.",
            "Hospitalization may be needed for severe cases.",
        ], [2, 1, 0, 1, 0, 2, 0, 1, 0, 1]),

        ("side effects of ibuprofen", [
            "Stomach irritation and heartburn.",
            "May increase risk of cardiovascular events.",
            "Pizza is an Italian dish.",
            "Dizziness and headache.",
            "The Eiffel Tower is in Paris.",
            "Kidney damage with long-term use.",
            "Climate change affects temperatures.",
            "Allergic reactions in some individuals.",
            "Shakespeare wrote plays.",
            "Take with food to reduce stomach upset.",
        ], [2, 1, 0, 1, 0, 2, 0, 1, 0, 1]),

        ("What is asthma?", [
            "Asthma causes airway inflammation and narrowing.",
            "Triggers include allergens exercise and cold air.",
            "Pizza is an Italian dish.",
            "Inhalers provide quick relief during attacks.",
            "The Eiffel Tower is in Paris.",
            "Symptoms include wheezing and shortness of breath.",
            "Climate change affects temperatures.",
            "Long-control medications prevent symptoms.",
            "Shakespeare wrote plays.",
            "Peak flow meter monitors lung function.",
        ], [2, 1, 0, 1, 0, 2, 0, 1, 0, 1]),

        # === LEGAL (6 queries) ===
        ("What is intellectual property?", [
            "IP protects creations of the mind.",
            "Patents protect inventions for 20 years.",
            "Pizza is an Italian dish.",
            "Copyright protects original works of authorship.",
            "The Eiffel Tower is in Paris.",
            "Trademarks protect brand names and logos.",
            "Climate change affects temperatures.",
            "Trade secrets protect confidential business information.",
            "Shakespeare wrote plays.",
            "IP infringement can result in civil penalties.",
        ], [2, 1, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("How to file a trademark?", [
            "Search USPTO database for existing marks.",
            "File application with USPTO including specimen.",
            "Pizza is an Italian dish.",
            "Registration takes 6-12 months.",
            "The Eiffel Tower is in Paris.",
            "Use trademark in commerce for protection.",
            "Climate change affects temperatures.",
            "Hire trademark attorney for complex cases.",
            "Shakespeare wrote plays.",
            "Renew trademark every 10 years.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("What is GDPR compliance?", [
            "GDPR protects EU citizens personal data.",
            "Requires explicit consent for data collection.",
            "Pizza is an Italian dish.",
            "Right to access and delete personal data.",
            "The Eiffel Tower is in Paris.",
            "Data breach notification within 72 hours.",
            "Climate change affects temperatures.",
            "Appoint Data Protection Officer for large processors.",
            "Shakespeare wrote plays.",
            "Fines up to 4% of global revenue for violations.",
        ], [2, 1, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("How to form an LLC?", [
            "Choose unique business name in your state.",
            "File Articles of Organization with Secretary of State.",
            "Pizza is an Italian dish.",
            "Create Operating Agreement for multi-member LLCs.",
            "The Eiffel Tower is in Paris.",
            "Obtain EIN from IRS for tax purposes.",
            "Climate change affects temperatures.",
            "Register for state and local business licenses.",
            "Shakespeare wrote plays.",
            "Annual reports required in most states.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("What is a non-disclosure agreement?", [
            "NDA is a legally binding confidentiality contract.",
            "Protects confidential information shared between parties.",
            "Pizza is an Italian dish.",
            "Unilateral vs bilateral NDAs.",
            "The Eiffel Tower is in Paris.",
            "Duration of confidentiality obligations.",
            "Climate change affects temperatures.",
            "Remedies for breach include injunctions.",
            "Shakespeare wrote plays.",
            "Define what constitutes confidential information.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("How to file a small claims lawsuit?", [
            "File complaint in small claims court.",
            "Jurisdictional limit typically $5000-$10000.",
            "Pizza is an Italian dish.",
            "Serve defendant with court papers.",
            "The Eiffel Tower is in Paris.",
            "Attend hearing with evidence and witnesses.",
            "Climate change affects temperatures.",
            "Judgment enforced through wage garnishment.",
            "Shakespeare wrote plays.",
            "No attorney required in most jurisdictions.",
        ], [2, 1, 0, 1, 0, 1, 0, 1, 0, 1]),

        # === NEWS (6 queries) ===
        ("climate change latest news", [
            "IPCC report warns of accelerating climate impacts.",
            "Record heatwaves across Europe and Asia.",
            "Pizza is an Italian dish.",
            "Renewable energy adoption reaches new highs.",
            "The Eiffel Tower is in Paris.",
            "Carbon capture technology advances.",
            "Machine learning improves climate predictions.",
            "Sea level rise threatens coastal cities.",
            "Shakespeare wrote plays.",
            "Climate protests demand policy action.",
        ], [2, 1, 0, 1, 0, 1, 0, 2, 0, 1]),

        ("AI regulation news", [
            "EU AI Act establishes risk-based framework.",
            "US executive order on AI safety standards.",
            "Pizza is an Italian dish.",
            "China implements AI content labeling rules.",
            "The Eiffel Tower is in Paris.",
            "Tech companies face new compliance requirements.",
            "Climate change affects temperatures.",
            "AI liability laws under debate.",
            "Shakespeare wrote plays.",
            "International AI governance summit planned.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("stock market today", [
            "S&P 500 reaches new all-time high.",
            "Tech stocks lead market rally.",
            "Pizza is an Italian dish.",
            "Federal Reserve signals rate cuts.",
            "The Eiffel Tower is in Paris.",
            "Energy sector volatility continues.",
            "Climate change affects temperatures.",
            "Bond yields decline on economic concerns.",
            "Shakespeare wrote plays.",
            "Cryptocurrency market cap surpasses $2T.",
        ], [2, 1, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("space exploration news", [
            "SpaceX Starship completes orbital test flight.",
            "NASA Artemis program returns to the Moon.",
            "Pizza is an Italian dish.",
            "James Webb telescope discovers new exoplanets.",
            "The Eiffel Tower is in Paris.",
            "China builds Tiangong space station.",
            "Climate change affects temperatures.",
            "Mars sample return mission delayed.",
            "Shakespeare wrote plays.",
            "Private space tourism expands.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("cybersecurity breach news", [
            "Major ransomware attack hits healthcare systems.",
            "Zero-day vulnerability in popular software.",
            "Pizza is an Italian dish.",
            "Data breach exposes millions of records.",
            "The Eiffel Tower is in Paris.",
            "Phishing campaigns target remote workers.",
            "Climate change affects temperatures.",
            "Critical infrastructure security concerns.",
            "Shakespeare wrote plays.",
            "Ransom payments reach record highs.",
        ], [2, 1, 0, 2, 0, 1, 0, 1, 0, 1]),

        ("electric vehicle market", [
            "Tesla maintains EV market leadership.",
            "Chinese EV makers expand globally.",
            "Pizza is an Italian dish.",
            "Battery costs decline 90% in decade.",
            "The Eiffel Tower is in Paris.",
            "Charging infrastructure expansion accelerates.",
            "Climate change affects temperatures.",
            "Traditional automakers transition to electric.",
            "Shakespeare wrote plays.",
            "Solid-state batteries promise longer range.",
        ], [2, 1, 0, 1, 0, 1, 0, 1, 0, 1]),

        # === DOCUMENTATION (6 queries) ===
        ("How to install Python?", [
            "Download Python from python.org.",
            "Run installer and check 'Add to PATH'.",
            "Pizza is an Italian dish.",
            "Verify installation with 'python --version'.",
            "The Eiffel Tower is in Paris.",
            "Use pyenv for multiple Python versions.",
            "Climate change affects temperatures.",
            "pip installs packages from PyPI.",
            "Shakespeare wrote plays.",
            "Virtual environments isolate project dependencies.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("What is Docker?", [
            "Docker packages applications into containers.",
            "Containers share the host OS kernel.",
            "Pizza is an Italian dish.",
            "Dockerfile defines container build steps.",
            "The Eiffel Tower is in Paris.",
            "Docker Compose orchestrates multi-container apps.",
            "Climate change affects temperatures.",
            "Images are read-only templates for containers.",
            "Shakespeare wrote plays.",
            "Docker Hub hosts public container images.",
        ], [2, 1, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("How to use Git?", [
            "git init creates a new repository.",
            "git commit saves changes to local repo.",
            "Pizza is an Italian dish.",
            "git push uploads changes to remote.",
            "The Eiffel Tower is in Paris.",
            "git branch creates parallel development lines.",
            "Climate change affects temperatures.",
            "git merge combines branch changes.",
            "Shakespeare wrote plays.",
            "git pull fetches and merges remote changes.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("What is REST API?", [
            "REST uses HTTP methods GET POST PUT DELETE.",
            "Stateless communication between client and server.",
            "Pizza is an Italian dish.",
            "JSON is the standard data format.",
            "The Eiffel Tower is in Paris.",
            "Status codes indicate request result.",
            "Climate change affects temperatures.",
            "Endpoints represent resources.",
            "Shakespeare wrote plays.",
            "Authentication via tokens or API keys.",
        ], [2, 1, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("How to write unit tests?", [
            "Tests verify individual code units work correctly.",
            "pytest is the most popular Python testing framework.",
            "Pizza is an Italian dish.",
            "Use Arrange-Act-Assert pattern.",
            "The Eiffel Tower is in Paris.",
            "Mock external dependencies for isolation.",
            "Climate change affects temperatures.",
            "Code coverage measures test completeness.",
            "Shakespeare wrote plays.",
            "TDD writes tests before implementation.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),

        ("What is CI/CD?", [
            "CI automatically builds and tests code changes.",
            "CD deploys verified changes to production.",
            "Pizza is an Italian dish.",
            "GitHub Actions provides CI/CD workflows.",
            "The Eiffel Tower is in Paris.",
            "Jenkins is a popular CI/CD server.",
            "Climate change affects temperatures.",
            "Pipeline stages include build test deploy.",
            "Shakespeare wrote plays.",
            "Automated rollback on deployment failure.",
        ], [2, 2, 0, 1, 0, 1, 0, 1, 0, 1]),
    ]


def benchmark_quality(model_name, test_cases, k=10):
    """Run quality benchmark for a model."""
    reranker = Reranker(model_name, offline=True, show_download_progress=False)

    ndcg_scores = []
    mrr_scores = []
    recall_scores = []

    for query, docs, relevances in test_cases:
        results = reranker.rerank(query, docs)

        scores_with_relevance = []
        for i, res in enumerate(results):
            scores_with_relevance.append((res.score, relevances[res.index]))

        ndcg = compute_ndcg(scores_with_relevance, k)
        mrr = compute_mrr(scores_with_relevance)
        recall = compute_recall(scores_with_relevance, k)

        ndcg_scores.append(ndcg)
        mrr_scores.append(mrr)
        recall_scores.append(recall)

    reranker.close()

    n = len(test_cases)
    return {
        'ndcg_mean': sum(ndcg_scores) / n,
        'mrr_mean': sum(mrr_scores) / n,
        'recall_mean': sum(recall_scores) / n,
        'ndcg_p95': sorted(ndcg_scores)[int(0.05 * n)],  # 5th percentile
        'ndcg_scores': ndcg_scores,
    }


def main():
    parser = argparse.ArgumentParser(description='Expanded quality benchmark (50 queries)')
    parser.add_argument('--k', type=int, default=10, help='Top-k for metrics')
    args = parser.parse_args()

    test_cases = create_test_corpus()

    print("=" * 70)
    print("Expanded Quality Benchmark: FP32 vs INT8")
    print("=" * 70)
    print(f"Queries: {len(test_cases)}")
    print(f"Metrics: NDCG@{args.k}, MRR, Recall@{args.k}")
    print()

    # FP32
    print("--- FP32 ---")
    fp32 = benchmark_quality("jinaai/jina-reranker-v1-turbo-en", test_cases, args.k)
    print(f"  NDCG@{args.k} mean: {fp32['ndcg_mean']:.4f}")
    print(f"  NDCG@{args.k} P5: {fp32['ndcg_p95']:.4f}")
    print(f"  MRR mean: {fp32['mrr_mean']:.4f}")
    print(f"  Recall@{args.k} mean: {fp32['recall_mean']:.4f}")
    print()

    # INT8
    print("--- INT8 ---")
    int8 = benchmark_quality("jinaai/jina-reranker-v1-turbo-en-quantized", test_cases, args.k)
    print(f"  NDCG@{args.k} mean: {int8['ndcg_mean']:.4f}")
    print(f"  NDCG@{args.k} P5: {int8['ndcg_p95']:.4f}")
    print(f"  MRR mean: {int8['mrr_mean']:.4f}")
    print(f"  Recall@{args.k} mean: {int8['recall_mean']:.4f}")
    print()

    # Comparison
    ndcg_diff = int8['ndcg_mean'] - fp32['ndcg_mean']
    ndcg_diff_pct = (ndcg_diff / fp32['ndcg_mean'] * 100) if fp32['ndcg_mean'] > 0 else 0
    mrr_diff = int8['mrr_mean'] - fp32['mrr_mean']
    recall_diff = int8['recall_mean'] - fp32['recall_mean']

    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Metric':<20} | {'FP32':>10} | {'INT8':>10} | {'Diff':>10} | {'Diff %':>10}")
    print("-" * 65)
    print(f"{'NDCG@10 mean':<20} | {fp32['ndcg_mean']:>10.4f} | {int8['ndcg_mean']:>10.4f} | {ndcg_diff:>+10.4f} | {ndcg_diff_pct:>+9.2f}%")
    print(f"{'NDCG@10 P5':<20} | {fp32['ndcg_p95']:>10.4f} | {int8['ndcg_p95']:>10.4f} | {int8['ndcg_p95']-fp32['ndcg_p95']:>+10.4f} | {'':>10}")
    print(f"{'MRR mean':<20} | {fp32['mrr_mean']:>10.4f} | {int8['mrr_mean']:>10.4f} | {mrr_diff:>+10.4f} | {'':>10}")
    print(f"{'Recall@10 mean':<20} | {fp32['recall_mean']:>10.4f} | {int8['recall_mean']:>10.4f} | {recall_diff:>+10.4f} | {'':>10}")
    print()

    # Decision
    print("=" * 70)
    print("DECISION")
    print("=" * 70)
    print()
    print(f"NDCG@10 difference: {ndcg_diff_pct:+.2f}%")
    print()

    if abs(ndcg_diff_pct) < 1.0:
        print("=> INT8 is the recommended default based on the 50-query benchmark.")
        print("   FP32 remains available for maximum-quality / compatibility-sensitive workloads.")
    elif abs(ndcg_diff_pct) < 3.0:
        print("=> NDCG diff 1-3%: Keep FP32 as default, INT8 as fast option.")
    else:
        print("=> NDCG diff > 3%: FP32 strongly recommended.")
    print()

    # Markdown
    print("## Results (for markdown)")
    print()
    print("| Metric | FP32 | INT8 | Diff |")
    print("|--------|------|------|------|")
    print(f"| NDCG@10 mean | {fp32['ndcg_mean']:.4f} | {int8['ndcg_mean']:.4f} | {ndcg_diff:+.4f} |")
    print(f"| MRR mean | {fp32['mrr_mean']:.4f} | {int8['mrr_mean']:.4f} | {mrr_diff:+.4f} |")
    print(f"| Recall@10 mean | {fp32['recall_mean']:.4f} | {int8['recall_mean']:.4f} | {recall_diff:+.4f} |")


if __name__ == '__main__':
    main()
