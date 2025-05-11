# Moses SMT Setup and Usage Guide

This document provides a comprehensive guide on how to set up and use the Moses Statistical Machine Translation (SMT) system with our English-Polish translation project.

## Overview

Moses SMT is a statistical machine translation system that learns to translate from parallel corpora. It provides significant improvements over our existing word-for-word "brutal" translation approach. This guide covers how to set up Moses SMT using Docker, prepare your data, train the system, and integrate it with our existing translation framework.

## Installation with Docker

Using Docker simplifies the installation process, avoiding complex dependencies.


### Steps
1. Pull the Moses SMT Docker image:
   ```bash
   docker pull amake/moses-smt:base
   ```

2. Clone the repository for additional configuration files:
   ```bash
   git clone https://github.com/amake/moses-smt.git
   cd moses-smt
   ```

## Preparing the Corpus

Moses requires parallel corpus data in TMX (Translation Memory eXchange) format. We'll convert our Europarl corpus to this format.

### Directory Structure
```
moses-smt/
├── tmx-train/  # Most of your TMX files (90%)
├── tmx-tune/   # A smaller portion for tuning (10%)
└── ...
```

### Converting Parallel Corpus to TMX
```bash
python create_tmx.py corpus/europarl.en corpus/europarl.pl europarl.tmx
```

Then split the resulting TMX file into training and tuning sets:
```bash
# Using head/tail for simple splitting (90% training, 10% tuning)
cat europarl.tmx | grep -v "<?xml" | grep -v "<!DOCTYPE" | grep -v "<tmx" | grep -v "</tmx>" | grep -v "<header" | grep -v "</header>" | grep -v "<body>" | grep -v "</body>" > europarl_content.txt

# Count lines and calculate split
total_lines=$(wc -l europarl_content.txt | awk '{print $1}')
train_lines=$(echo "$total_lines * 0.9" | bc | awk '{print int($0)}')
tune_lines=$(echo "$total_lines - $train_lines" | bc)

# Create header and footer files
echo '<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE tmx SYSTEM "tmx14.dtd">
<tmx version="1.4">
  <header creationtool="SimpleTMXCreator" creationtoolversion="1.0" segtype="sentence" o-tmf="PlainText" adminlang="en" srclang="en" datatype="plaintext">
  </header>
  <body>' > header.txt

echo '  </body>
</tmx>' > footer.txt

# Split the content
head -n $train_lines europarl_content.txt > train_content.txt
tail -n $tune_lines europarl_content.txt > tune_content.txt

# Combine to create final files
cat header.txt train_content.txt footer.txt > moses-smt/tmx-train/europarl.tmx
cat header.txt tune_content.txt footer.txt > moses-smt/tmx-tune/europarl.tmx

# Clean up
rm header.txt footer.txt europarl_content.txt train_content.txt tune_content.txt
```

## Training the Moses SMT System

With the TMX files prepared, we can now train Moses:

```bash
cd moses-smt
make SOURCE_LANG=en TARGET_LANG=pl LABEL=europarl
```

The command:
- Takes the files in tmx-train and tmx-tune
- Extracts the text and aligns the sentences
- Trains the language and translation models
- Tunes the model weights
- Creates a Docker image ready to use

When complete, you'll have a Docker image tagged as `moses-smt:europarl-en-pl`.

## Running the Moses Server

To start the Moses server:

```bash
make server SOURCE_LANG=en TARGET_LANG=pl LABEL=europarl PORT=8080
```

This starts the Moses server, making it accessible via XML-RPC at http://localhost:8080/RPC2.

You can verify it's working with a simple Python test:

```python
import xmlrpc.client

server = xmlrpc.client.ServerProxy("http://localhost:8080/RPC2")
params = {"text": "This is a test."}
result = server.translate(params)
print(result['text'])
```

## Integrating with Our Translation System

Our translation system uses a `Translator` abstract base class. We've implemented a `MosesTranslator` class that connects to the Moses server.

```python
# Import and create the translator
from translations.models.moses_translator import MosesTranslator

translator = MosesTranslator("http://localhost:8080/RPC2")

# Translate a single text
translated = translator.translate("Hello, world!")
print(translated)

# Or use with a dataset
translated_dataset = translator.translate_batch(your_dataset)
```

## Evaluation and Fine-Tuning

### Evaluating Translation Quality

To evaluate the quality of Moses translations versus our word-for-word approach:

1. **BLEU Score Evaluation**:
   ```bash
   # In moses-smt directory
   mosesdecoder/scripts/generic/multi-bleu.perl reference.pl < translated.pl
   ```

2. **Manual Evaluation**:
   - Sample a subset of translations
   - Compare Moses output to our word-for-word translations and reference translations
   - Assess fluency, adequacy, and grammatical correctness
