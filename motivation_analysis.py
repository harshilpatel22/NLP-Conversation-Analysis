"""
Mentor Motivation Analysis: Intrinsic vs Extrinsic Classifier
Optimized for 200K rows with multilingual support (EN, ES, AR, RU, ID)
Uses Aya Expanse 32B (quantized) for 80GB GPU

╔════════════════════════════════════════════════════════════════════╗
║                    GOOGLE COLAB SETUP GUIDE                        ║
╚════════════════════════════════════════════════════════════════════╝

✨ NEW FEATURES (Updated):
  • 8K token context window (increased from 512 tokens)
  • Smart language diversity sampling
  • Uses existing 'detected_language' column from your dataset
  • Stratified sampling across target languages (EN, ES, AR, RU, ID)
  • Language-specific analysis in results

CELL 1: Mount Google Drive (CRITICAL - Run this first!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This saves the model (~20GB) to your Google Drive so you NEVER
have to redownload it again on future runs!

    from google.colab import drive
    drive.mount('/content/drive')

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CELL 2: Install Dependencies
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    !pip install transformers torch accelerate bitsandbytes pandas numpy tqdm

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CELL 3: Upload your CSV and run this entire script
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Upload your CSV file using the file upload button
- Update CSV_PATH variable at the bottom of this script
- Run the entire cell

SAMPLING STRATEGY:
- First filters to rows with 50+ characters
- Uses existing 'detected_language' column from your dataset
- Samples 100 rows with diverse language representation
- Prioritizes target languages: English, Spanish, Arabic, Russian, Indonesian
- Ensures balanced representation across languages

ANALYSIS OUTPUT:
- motivation_type: INTRINSIC or EXTRINSIC
- confidence: 0.0 to 1.0
- intrinsic_extrinsic_ratio: 1.0 (intrinsic) to 0.0 (extrinsic)
- detected_language: From your existing column
- reasoning: How classification was determined
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

First run: Downloads model to Drive (~20GB, takes 10-15 min)
Future runs: Loads instantly from Drive cache! ⚡
"""

# ===== INSTALLATION (Run first) =====
# !pip install transformers torch accelerate bitsandbytes pandas numpy tqdm

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import re
from typing import Optional, Tuple
import json
import os
from pathlib import Path
from collections import Counter

# ===== MOUNT GOOGLE DRIVE (Run this first to persist model cache) =====
def mount_google_drive():
    """Mount Google Drive to persist model downloads across sessions"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted successfully!")
        return True
    except:
        print("⚠ Not running in Colab or Drive already mounted")
        return False


def get_directory_size(path):
    """Calculate total size of a directory in GB"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size / (1024**3)  # Convert to GB
    except:
        return 0


def check_model_cache():
    """Check if model is cached and display cache info"""
    print("\n" + "="*60)
    print("MODEL CACHE STATUS")
    print("="*60)

    if os.path.exists(Config.CACHE_DIR):
        cache_size = get_directory_size(Config.CACHE_DIR)
        tokenizer_path = os.path.join(Config.CACHE_DIR, "tokenizer")
        model_path = os.path.join(Config.CACHE_DIR, "model")

        tokenizer_exists = os.path.exists(tokenizer_path)
        model_exists = os.path.exists(model_path)

        print(f"Cache directory: {Config.CACHE_DIR}")
        print(f"Cache size: {cache_size:.2f} GB")
        print(f"Tokenizer cached: {'✓ YES' if tokenizer_exists else '✗ NO'}")
        print(f"Model cached: {'✓ YES' if model_exists else '✗ NO'}")

        if tokenizer_exists and model_exists:
            print("\n🎉 Model fully cached! Loading will be instant.")
        else:
            print("\n⚠ Model not fully cached. First run will download ~20GB.")
    else:
        print(f"Cache directory: {Config.CACHE_DIR}")
        print("Status: ✗ Cache directory doesn't exist yet")
        print("\n⚠ First run will download model (~20GB, 10-15 min)")

    print("="*60 + "\n")

# ===== CONFIGURATION =====
class Config:
    # Model cache directory (persists across Colab sessions when using Drive)
    # OPTION 1: Google Drive (RECOMMENDED - persists forever)
    CACHE_DIR = "/content/drive/MyDrive/model_cache/aya_expanse_32b"

    # OPTION 2: Local Colab storage (WARNING: deleted after session ends)
    # CACHE_DIR = "/content/model_cache/aya_expanse_32b"

    # OPTION 3: Custom path (if you have a specific Drive folder structure)
    # CACHE_DIR = "/content/drive/MyDrive/YourFolder/models/aya_expanse_32b"

    # Model settings
    MODEL_NAME = "CohereForAI/aya-expanse-32b"  # Multilingual powerhouse
    MIN_CHARS = 50  # Only analyze text with 50+ chars
    MAX_LENGTH = 8000  # Max token length for model input (increased to 8k)

    # Sampling settings
    TEST_SAMPLE_SIZE = 100  # For testing
    FULL_SAMPLE_SIZE = None  # None = all rows, or set integer for sampling
    ENSURE_LANGUAGE_DIVERSITY = True  # Sample across different languages
    TARGET_LANGUAGES = ['en', 'es', 'ar', 'ru']  # Priority languages (removed 'id')
    LANGUAGE_DISTRIBUTION = {  # Target distribution for sampling
        'en': 0.70,  # English 70%
        'es': 0.20,  # Spanish 20%
        'ar': 0.05,  # Arabic 5%
        'ru': 0.05   # Russian 5%
    }

    # GPU settings
    USE_4BIT = True  # 4-bit quantization for 80GB GPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Batch processing
    BATCH_SIZE = 8  # Adjust based on GPU memory

    # Debug settings
    VERBOSE = True  # Set to True to see model responses (helpful for debugging)
    SHOW_FIRST_N_RESPONSES = 5  # Show first N raw responses when verbose


# ===== STEP 1: LOAD AND PREPARE DATA =====
def load_and_prepare_data(filepath: str, test_mode: bool = True) -> pd.DataFrame:
    """Load CSV and extract mentor motivation column with diverse language sampling"""
    print("\n" + "="*60)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("="*60)

    print("\nLoading dataset...")
    df = pd.read_csv(filepath)
    print(f"✓ Total rows loaded: {len(df):,}")

    # Create mentor_motivation column based on who is the mentor
    df['mentor_motivation'] = df.apply(
        lambda row: row['person1_motivation_mentorship']
        if str(row['person1_is_mentor']).lower() == 'true'
        else row['person2_motivation_mentorship'],
        axis=1
    )

    # Filter: only rows with sufficient text length
    df['text_length'] = df['mentor_motivation'].fillna('').astype(str).str.len()
    df_valid = df[df['text_length'] >= Config.MIN_CHARS].copy()

    print(f"\n✓ Rows with {Config.MIN_CHARS}+ characters: {len(df_valid):,} ({len(df_valid)/len(df)*100:.1f}%)")

    # Check if detected_language column already exists
    if 'detected_language' not in df_valid.columns:
        print("\n⚠ Warning: 'detected_language' column not found in dataset")
        print("   Creating placeholder 'unknown' language labels")
        df_valid['detected_language'] = 'unknown'
    else:
        print("\n✓ Using existing 'detected_language' column from dataset")

    # Language distribution
    lang_dist = df_valid['detected_language'].value_counts()
    print("\n--- Language Distribution in Dataset ---")
    for lang, count in lang_dist.head(10).items():
        lang_name = {
            'en': 'English', 'es': 'Spanish', 'ar': 'Arabic',
            'ru': 'Russian', 'id': 'Indonesian', 'pt': 'Portuguese',
            'fr': 'French', 'de': 'German', 'it': 'Italian',
            'unknown': 'Unknown'
        }.get(lang, lang)
        print(f"  {lang_name:15} ({lang}): {count:5,} ({count/len(df_valid)*100:5.1f}%)")

    # Smart sampling with language diversity
    if test_mode:
        sample_size = min(Config.TEST_SAMPLE_SIZE, len(df_valid))

        if Config.ENSURE_LANGUAGE_DIVERSITY:
            print(f"\n🎯 Smart sampling: {sample_size} rows with diverse languages...")
            df_sampled = diverse_language_sample(df_valid, sample_size)
        else:
            print(f"\n📊 Random sampling: {sample_size} rows...")
            df_sampled = df_valid.sample(n=sample_size, random_state=42)

        print(f"✓ Test mode: Analyzing {len(df_sampled)} samples")
    else:
        if Config.FULL_SAMPLE_SIZE and Config.FULL_SAMPLE_SIZE < len(df_valid):
            print(f"\n📊 Sampling {Config.FULL_SAMPLE_SIZE:,} rows...")
            if Config.ENSURE_LANGUAGE_DIVERSITY:
                df_sampled = diverse_language_sample(df_valid, Config.FULL_SAMPLE_SIZE)
            else:
                df_sampled = df_valid.sample(n=Config.FULL_SAMPLE_SIZE, random_state=42)
        else:
            df_sampled = df_valid
            print(f"\n✓ Processing all {len(df_valid):,} valid rows")

    # Show sample language distribution
    if len(df_sampled) < len(df_valid):
        sample_lang_dist = df_sampled['detected_language'].value_counts()
        print("\n--- Language Distribution in Sample ---")
        for lang, count in sample_lang_dist.items():
            lang_name = {
                'en': 'English', 'es': 'Spanish', 'ar': 'Arabic',
                'ru': 'Russian', 'id': 'Indonesian', 'pt': 'Portuguese',
                'fr': 'French', 'de': 'German', 'it': 'Italian',
                'unknown': 'Unknown'
            }.get(lang, lang)
            print(f"  {lang_name:15} ({lang}): {count:3} ({count/len(df_sampled)*100:5.1f}%)")

    print("\n" + "="*60)
    return df_sampled.reset_index(drop=True)


def diverse_language_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """
    Sample rows with specific language distribution:
    English 70%, Spanish 20%, Arabic 5%, Russian 5%
    """
    # Count languages
    lang_counts = df['detected_language'].value_counts()

    samples = []

    print(f"\n  Target distribution: EN 70%, ES 20%, AR 5%, RU 5%")

    for lang, target_pct in Config.LANGUAGE_DISTRIBUTION.items():
        if lang in lang_counts.index:
            lang_df = df[df['detected_language'] == lang]
            n_samples = int(sample_size * target_pct)
            n_samples = min(n_samples, len(lang_df))  # Can't sample more than available

            if n_samples > 0:
                samples.append(lang_df.sample(n=n_samples, random_state=42))
                lang_name = {'en': 'English', 'es': 'Spanish', 'ar': 'Arabic', 'ru': 'Russian'}.get(lang, lang)
                print(f"    ✓ {lang_name:10} ({lang}): {n_samples} samples")

    # Combine samples
    result = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()

    # If we don't have enough, fill with random samples from any language
    if len(result) < sample_size:
        remaining_df = df[~df.index.isin(result.index)]
        if len(remaining_df) > 0:
            additional = min(sample_size - len(result), len(remaining_df))
            result = pd.concat([result, remaining_df.sample(n=additional, random_state=42)], ignore_index=True)
            print(f"    ✓ Other languages: {additional} samples (to fill remaining)")

    # If we have too many, trim randomly
    if len(result) > sample_size:
        result = result.sample(n=sample_size, random_state=42)

    return result


# ===== STEP 2: LOAD QUANTIZED MODEL =====
def load_quantized_model():
    """Load Aya Expanse 32B with 4-bit quantization for 80GB GPU"""

    print("\n" + "="*60)
    print("STEP 2: LOADING MODEL")
    print("="*60)

    # Create cache directory if it doesn't exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    print(f"\nModel cache directory: {Config.CACHE_DIR}")

    # Check if model is already cached
    tokenizer_path = os.path.join(Config.CACHE_DIR, "tokenizer")
    model_path = os.path.join(Config.CACHE_DIR, "model")

    is_cached = os.path.exists(tokenizer_path) and os.path.exists(model_path)

    if is_cached:
        print("✓ Model found in cache! Loading from local storage...")
    else:
        print(f"⚠ Model not found in cache. Downloading {Config.MODEL_NAME}...")
        print("   This is a one-time download (~20GB). Future runs will be instant!")

    print(f"\nLoading {Config.MODEL_NAME} with 4-bit quantization...")
    print(f"Context length: {Config.MAX_LENGTH:,} tokens")

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer with cache
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME,
        cache_dir=Config.CACHE_DIR,
        trust_remote_code=True,
        model_max_length=Config.MAX_LENGTH  # Set max length
    )

    # Save tokenizer to specific path for easier checking
    if not is_cached:
        tokenizer.save_pretrained(tokenizer_path)
        print(f"✓ Tokenizer saved to cache: {tokenizer_path}")

    # Load model with quantization and cache
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        cache_dir=Config.CACHE_DIR,
        max_position_embeddings=Config.MAX_LENGTH  # Support 8k context
    )

    # Save model to specific path for easier checking
    if not is_cached:
        model.save_pretrained(model_path)
        print(f"✓ Model saved to cache: {model_path}")

    model.eval()
    print("✓ Model loaded successfully!")
    print(f"Model memory footprint: ~{model.get_memory_footprint() / 1e9:.2f} GB")
    print("="*60)

    return model, tokenizer


# ===== STEP 3: CREATE CLASSIFICATION PROMPT =====
def create_classification_prompt(motivation_text: str) -> str:
    """
    Create a prompt with few-shot examples for better classification
    """
    prompt = f"""You are an expert in analyzing mentorship motivations. Classify the mentor's motivation as either INTRINSIC or EXTRINSIC.

INTRINSIC motivation = focused on SELF (learning, personal growth, own skills, own satisfaction)
EXTRINSIC motivation = focused on OTHERS (helping mentee, mentee's growth, giving back, community impact)

Examples:

Text: "I want to learn from experienced entrepreneurs and develop my own business skills"
Answer: INTRINSIC

Text: "I want to help others succeed and share my knowledge to make an impact"
Answer: EXTRINSIC

Text: "To give back to the community and support aspiring entrepreneurs in their journey"
Answer: EXTRINSIC

Text: "Personal development and networking to grow my own expertise"
Answer: INTRINSIC

Text: "I'm passionate about mentoring others and seeing them achieve their goals"
Answer: EXTRINSIC

Now classify this motivation:

Text: "{motivation_text}"
Answer:"""

    return prompt


# ===== STEP 4: CLASSIFY SINGLE TEXT =====
def classify_motivation(
    text: str,
    model,
    tokenizer,
    show_response: bool = False
) -> Tuple[str, float, str]:
    """Classify a single motivation text with robust parsing"""

    prompt = create_classification_prompt(text)

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=Config.MAX_LENGTH
    ).to(Config.DEVICE)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Shorter - we just need INTRINSIC or EXTRINSIC
            temperature=0.1,  # Very low for consistent answers
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the answer part (after the prompt)
    if "Answer:" in full_response:
        response = full_response.split("Answer:")[-1].strip()
    else:
        response = full_response[len(prompt):].strip()

    if show_response:
        print(f"\n--- Raw Model Response ---")
        print(f"Full response length: {len(full_response)} chars")
        print(f"Extracted answer: '{response[:200]}'")
        print(f"-------------------------")

    # Multi-strategy parsing
    classification = 'UNKNOWN'
    confidence = 0.5
    reasoning = 'Parsed from response'

    # Strategy 1: Direct keyword matching (most reliable)
    response_upper = response.upper()

    if 'INTRINSIC' in response_upper:
        classification = 'INTRINSIC'
        # Check if answer is clear and at the beginning
        if response_upper.startswith('INTRINSIC'):
            confidence = 0.95
        else:
            confidence = 0.85
    elif 'EXTRINSIC' in response_upper:
        classification = 'EXTRINSIC'
        if response_upper.startswith('EXTRINSIC'):
            confidence = 0.95
        else:
            confidence = 0.85

    # Strategy 2: Keyword-based classification as fallback
    if classification == 'UNKNOWN':
        intrinsic_keywords = [
            'learn', 'personal growth', 'develop my', 'improve my', 'own skills',
            'networking', 'experience', 'grow my', 'my development', 'self'
        ]
        extrinsic_keywords = [
            'help others', 'support', 'give back', 'community', 'mentee',
            'guide', 'assist', 'teach', 'share knowledge', 'impact on others',
            'helping', 'contribute'
        ]

        text_lower = text.lower()
        intrinsic_count = sum(1 for kw in intrinsic_keywords if kw in text_lower)
        extrinsic_count = sum(1 for kw in extrinsic_keywords if kw in text_lower)

        if intrinsic_count > extrinsic_count and intrinsic_count > 0:
            classification = 'INTRINSIC'
            confidence = min(0.6 + (intrinsic_count * 0.05), 0.8)
            reasoning = f'Keyword-based (intrinsic: {intrinsic_count}, extrinsic: {extrinsic_count})'
        elif extrinsic_count > intrinsic_count and extrinsic_count > 0:
            classification = 'EXTRINSIC'
            confidence = min(0.6 + (extrinsic_count * 0.05), 0.8)
            reasoning = f'Keyword-based (intrinsic: {intrinsic_count}, extrinsic: {extrinsic_count})'
        else:
            # Neutral - default to extrinsic (mentors usually focus on helping)
            classification = 'EXTRINSIC'
            confidence = 0.5
            reasoning = 'Default classification (ambiguous)'

    return classification, confidence, reasoning


# ===== STEP 5: BATCH PROCESSING =====
def process_batch(df: pd.DataFrame, model, tokenizer) -> pd.DataFrame:
    """Process all rows and add classification results"""

    print("\n" + "="*60)
    print("STEP 3: PROCESSING BATCH")
    print("="*60)

    print(f"\nProcessing {len(df)} rows...")
    if Config.VERBOSE:
        print(f"(Showing first {Config.SHOW_FIRST_N_RESPONSES} raw responses for debugging)")

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing"):
        motivation_text = str(row['mentor_motivation'])
        detected_lang = row.get('detected_language', 'unknown')

        # Show raw responses for first N items if verbose mode is on
        show_response = Config.VERBOSE and len(results) < Config.SHOW_FIRST_N_RESPONSES

        try:
            classification, confidence, reasoning = classify_motivation(
                motivation_text,
                model,
                tokenizer,
                show_response=show_response
            )
        except Exception as e:
            print(f"\n⚠ Error processing row {idx}: {e}")
            classification = 'ERROR'
            confidence = 0.0
            reasoning = str(e)

        results.append({
            'classification': classification,
            'confidence': confidence,
            'reasoning': reasoning
        })

        # Show sample progress
        if show_response:
            lang_name = {
                'en': 'English', 'es': 'Spanish', 'ar': 'Arabic',
                'ru': 'Russian', 'id': 'Indonesian'
            }.get(detected_lang, detected_lang)
            print(f"\nRow {idx} [{lang_name}]: '{motivation_text[:100]}...'")
            print(f"→ Result: {classification} (confidence: {confidence:.2f})")

    # Add results to dataframe
    df['motivation_type'] = [r['classification'] for r in results]
    df['confidence'] = [r['confidence'] for r in results]
    df['reasoning'] = [r['reasoning'] for r in results]

    # Calculate intrinsic/extrinsic ratio (1.0 = fully intrinsic, 0.0 = fully extrinsic)
    df['intrinsic_extrinsic_ratio'] = df.apply(
        lambda row: row['confidence'] if row['motivation_type'] == 'INTRINSIC'
        else (1 - row['confidence']) if row['motivation_type'] == 'EXTRINSIC'
        else 0.5,  # Neutral for UNKNOWN
        axis=1
    )

    print("\n✓ Batch processing complete!")
    print("="*60)

    return df


# ===== STEP 6: ANALYZE RESULTS =====
def analyze_results(df: pd.DataFrame):
    """Print summary statistics of the analysis"""

    print("\n" + "="*60)
    print("STEP 4: ANALYSIS RESULTS")
    print("="*60)

    print(f"\n📊 Total analyzed: {len(df)}")

    # Language breakdown
    if 'detected_language' in df.columns:
        print("\n--- Language Distribution ---")
        lang_dist = df['detected_language'].value_counts()
        for lang, count in lang_dist.items():
            lang_name = {
                'en': 'English', 'es': 'Spanish', 'ar': 'Arabic',
                'ru': 'Russian', 'id': 'Indonesian', 'pt': 'Portuguese',
                'fr': 'French', 'de': 'German', 'unknown': 'Unknown'
            }.get(lang, lang)
            print(f"  {lang_name:15} ({lang}): {count:3} ({count/len(df)*100:5.1f}%)")

    # Classification distribution
    print("\n--- Classification Distribution ---")
    print(df['motivation_type'].value_counts())

    intrinsic_pct = (df['motivation_type']=='INTRINSIC').sum()/len(df)*100
    extrinsic_pct = (df['motivation_type']=='EXTRINSIC').sum()/len(df)*100
    unknown_pct = (df['motivation_type']=='UNKNOWN').sum()/len(df)*100

    print(f"\n  Intrinsic: {intrinsic_pct:.1f}%")
    print(f"  Extrinsic: {extrinsic_pct:.1f}%")
    print(f"  Unknown: {unknown_pct:.1f}%")

    # Classification by language
    if 'detected_language' in df.columns:
        print("\n--- Classification by Language ---")
        for lang in df['detected_language'].unique():
            lang_df = df[df['detected_language'] == lang]
            if len(lang_df) > 0:
                lang_name = {
                    'en': 'English', 'es': 'Spanish', 'ar': 'Arabic',
                    'ru': 'Russian', 'id': 'Indonesian'
                }.get(lang, lang)
                intrinsic = (lang_df['motivation_type'] == 'INTRINSIC').sum()
                extrinsic = (lang_df['motivation_type'] == 'EXTRINSIC').sum()
                unknown = (lang_df['motivation_type'] == 'UNKNOWN').sum()
                print(f"  {lang_name:15}: I:{intrinsic:2} E:{extrinsic:2} U:{unknown:2} (total:{len(lang_df):2})")

    # Reasoning breakdown
    print("\n--- Classification Method Breakdown ---")
    reasoning_summary = df['reasoning'].value_counts()
    for reason, count in reasoning_summary.items():
        print(f"  {reason}: {count} ({count/len(df)*100:.1f}%)")

    # Confidence statistics (excluding unknowns)
    classified_df = df[df['motivation_type'].isin(['INTRINSIC', 'EXTRINSIC'])]
    if len(classified_df) > 0:
        print("\n--- Confidence Statistics (Classified Only) ---")
        print(f"  Mean confidence: {classified_df['confidence'].mean():.3f}")
        print(f"  Median confidence: {classified_df['confidence'].median():.3f}")
        print(f"  Min confidence: {classified_df['confidence'].min():.3f}")
        print(f"  Max confidence: {classified_df['confidence'].max():.3f}")

        # Intrinsic/Extrinsic ratio statistics
        print("\n--- Intrinsic/Extrinsic Ratio Statistics ---")
        print(f"  Mean ratio: {classified_df['intrinsic_extrinsic_ratio'].mean():.3f}")
        print(f"  Median ratio: {classified_df['intrinsic_extrinsic_ratio'].median():.3f}")
        print(f"  Std dev: {classified_df['intrinsic_extrinsic_ratio'].std():.3f}")

        # Examples
        print("\n--- Sample Classifications ---")
        for motivation_type in ['INTRINSIC', 'EXTRINSIC']:
            type_df = df[df['motivation_type'] == motivation_type]
            if len(type_df) > 0:
                print(f"\n{motivation_type} Examples ({len(type_df)} total):")
                samples = type_df.head(3)
                for idx, row in samples.iterrows():
                    lang = row.get('detected_language', 'unknown')
                    lang_name = {
                        'en': 'EN', 'es': 'ES', 'ar': 'AR',
                        'ru': 'RU', 'id': 'ID'
                    }.get(lang, lang.upper())
                    print(f"\n  [{lang_name}] {row['mentor_motivation'][:120]}...")
                    print(f"      → Ratio: {row['intrinsic_extrinsic_ratio']:.3f} | Confidence: {row['confidence']:.3f}")
                    print(f"      → Method: {row['reasoning']}")
    else:
        print("\n⚠ No successful classifications found!")
        print("\nDEBUG INFO: Check the raw responses above to see what the model is generating.")

    print("\n" + "="*60)


# ===== MAIN EXECUTION =====
def main(csv_path: str, test_mode: bool = True, output_path: Optional[str] = None):
    """Main execution function"""

    print("\n" + "█"*60)
    print("  MENTOR MOTIVATION NLP ANALYSIS")
    print("  Intrinsic vs Extrinsic Classification")
    print("█"*60)
    print(f"\n🖥️  Device: {Config.DEVICE}")
    print(f"📏 Context length: {Config.MAX_LENGTH:,} tokens")
    print(f"🌍 Language diversity sampling: {'✓ Enabled' if Config.ENSURE_LANGUAGE_DIVERSITY else '✗ Disabled'}")

    # Check model cache status
    check_model_cache()

    # Step 1: Load and prepare data with diverse language sampling
    df = load_and_prepare_data(csv_path, test_mode=test_mode)

    # Step 2: Load model (will use cache if available)
    model, tokenizer = load_quantized_model()

    # Step 3: Process data
    df_results = process_batch(df, model, tokenizer)

    # Step 4: Analyze results
    analyze_results(df_results)

    # Step 5: Save results
    if output_path:
        # Include language info in output
        df_results.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")
        print(f"   Columns: {', '.join(df_results.columns.tolist())}")

    # Troubleshooting tips
    unknown_pct = (df_results['motivation_type']=='UNKNOWN').sum() / len(df_results) * 100
    if unknown_pct > 50:
        print("\n" + "="*60)
        print("⚠ TROUBLESHOOTING: High UNKNOWN rate detected!")
        print("="*60)
        print("\nPossible solutions:")
        print("1. Check the raw responses above - is the model generating text?")
        print("2. Try a different model (e.g., 'CohereForAI/aya-expanse-8b')")
        print("3. Increase temperature in classify_motivation() to 0.5")
        print("4. The keyword-based fallback should work - check if texts are in expected languages")
        print("5. Try with English-only samples first to test")
        print("="*60)

    return df_results


# ===== RUN THE ANALYSIS =====
if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  MENTOR MOTIVATION ANALYSIS - SETUP INSTRUCTIONS          ║
    ╚════════════════════════════════════════════════════════════╝

    ✅ NEW FEATURES:
    • 8K token context length (increased from 512)
    • Smart language diversity sampling
    • Uses existing 'detected_language' column from your dataset
    • Stratified sampling across EN, ES, AR, RU, ID

    STEP 1: Mount Google Drive (IMPORTANT - saves model for reuse!)
    ---------------------------------------------------------------
    Run this in a separate cell FIRST:

        from google.colab import drive
        drive.mount('/content/drive')

    This saves the 20GB model to your Drive so you never have to
    redownload it again!

    STEP 2: Install dependencies
    ---------------------------------------------------------------
    Run this in a separate cell:

        !pip install transformers torch accelerate bitsandbytes pandas numpy tqdm

    STEP 3: Upload your CSV file to Colab
    ---------------------------------------------------------------

    STEP 4: Update the CSV_PATH below and run this entire script
    ---------------------------------------------------------------
    """)

    # Upload your CSV file to Colab first, then specify the path
    CSV_PATH = "finaldata.csv"  # ⬅️ CHANGE THIS TO YOUR FILE PATH

    # TEST MODE: Run on 100 diverse language samples first
    print("\n" + "█"*60)
    print("  RUNNING TEST MODE")
    print("  Analyzing 100 samples with diverse languages")
    print("█"*60)

    df_test = main(
        csv_path=CSV_PATH,
        test_mode=True,
        output_path="test_results.csv"
    )

    print("\n" + "="*60)
    print("✓ TEST COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review test_results.csv")
    print("2. Check classification quality")
    print("3. If satisfied, uncomment the FULL ANALYSIS section below")
    print("="*60)

    # Uncomment below to run on full dataset (or sampled subset)
    # print("\n\n" + "█"*60)
    # print("  RUNNING FULL ANALYSIS")
    # print("█"*60)
    # df_full = main(
    #     csv_path=CSV_PATH,
    #     test_mode=False,
    #     output_path="full_results.csv"
    # )