import streamlit as st
import re
from collections import Counter
from functools import lru_cache
from typing import List, Tuple, Set, Dict

class AutocorrectTool:
    def __init__(self):
        self.vocab = set()
        self.word_freq = Counter()
        self.max_edit_distance = 2
        self.top_suggestions = 5
        self.common_phrases = {
            "i name was": "my name is",
            "i name is": "my name is",
            "i was born": "I was born",
            "i live in": "I live in",
            "i am from": "I am from",
            "i have a": "I have a"
        }
        self.proper_nouns = set()
        self.tense_pairs = {  # Added tense correction rules
            'was': 'is',
            'is': 'was',
            'were': 'are',
            'are': 'were',
            'has': 'had',
            'had': 'has',
            'go': 'went',
            'went': 'go'
        }
        self.time_indicators = {  # Words that indicate past/future tense
            'past': ['yesterday', 'ago', 'last', 'before', 'earlier'],
            'present': ['today', 'now', 'currently'],
            'future': ['tomorrow', 'next', 'will', 'soon', 'later']
        }
        
    def build_vocabulary(self, corpus: List[str]) -> None:
        words = re.findall(r'\w+', ' '.join(corpus).lower())
        self.vocab = set(words)
        self.word_freq = Counter(words)
        
        common_words = {
            'i': 10000, 'my': 5000, 'was': 5000, 'is': 5000, 
            'the': 10000, 'and': 8000, 'you': 6000, 'he': 5000,
            'she': 5000, 'we': 5000, 'they': 5000, 'it': 5000
        }
        for word, freq in common_words.items():
            self.vocab.add(word)
            self.word_freq[word] += freq
            
    @lru_cache(maxsize=10000)
    def _edit_distance(self, word1: str, word2: str) -> int:
        if not word1: return len(word2)
        if not word2: return len(word1)
        if word1[0] == word2[0]:
            return self._edit_distance(word1[1:], word2[1:])
        return 1 + min(
            self._edit_distance(word1[1:], word2),
            self._edit_distance(word1, word2[1:]),
            self._edit_distance(word1[1:], word2[1:])
        )
    
    def _get_candidates(self, word: str) -> List[str]:
        word_lower = word.lower()
        if word_lower in self.vocab:
            return [word]
        if word_lower == 'i':
            return ['I']
            
        candidates = set()
        for i in range(1, min(self.max_edit_distance + 1, len(word))):
            for candidate in self._generate_edits(word_lower, i):
                if candidate in self.vocab:
                    candidates.add(candidate)
        if not candidates:
            return [word]
        return sorted(candidates, 
                    key=lambda x: (-self.word_freq[x], 
                                self._edit_distance(word_lower, x)))[:self.top_suggestions]
    
    def _generate_edits(self, word: str, distance: int) -> Set[str]:
        if distance == 0:
            return {word}
        edits = set()
        for edit in self._generate_edits(word, distance - 1):
            for i in range(len(edit)):
                edits.add(edit[:i] + edit[i+1:])
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    edits.add(edit[:i] + c + edit[i:])
                    edits.add(edit[:i] + c + edit[i+1:])
            for i in range(len(edit)-1):
                edits.add(edit[:i] + edit[i+1] + edit[i] + edit[i+2:])
        return edits
    
    def detect_tense_context(self, sentence: str) -> str:
        """Determine if sentence is past/present/future tense"""
        words = sentence.lower().split()
        for word in words:
            if word in self.time_indicators['past']:
                return 'past'
            if word in self.time_indicators['present']:
                return 'present'
            if word in self.time_indicators['future']:
                return 'future'
        return 'unknown'
    
    def correct_tense(self, sentence: str) -> str:
        """Correct verb tenses based on context"""
        tense_context = self.detect_tense_context(sentence)
        words = sentence.split()
        corrected = []
        
        for i, word in enumerate(words):
            lower_word = word.lower()
            if lower_word in self.tense_pairs:
                # Check if tense needs correction based on context
                if (tense_context == 'past' and lower_word in ['is', 'are', 'has']) or \
                   (tense_context == 'present' and lower_word in ['was', 'were', 'had']):
                    corrected_word = self.tense_pairs[lower_word]
                    # Preserve capitalization
                    if word[0].isupper():
                        corrected_word = corrected_word[0].upper() + corrected_word[1:]
                    corrected.append(corrected_word)
                else:
                    corrected.append(word)
            else:
                corrected.append(word)
                
        return ' '.join(corrected)
    
    def correct_phrases(self, text: str) -> str:
        text_lower = text.lower()
        for phrase, correction in self.common_phrases.items():
            if phrase in text_lower:
                start_idx = text_lower.find(phrase)
                if start_idx != -1:
                    end_idx = start_idx + len(phrase)
                    text = text[:start_idx] + correction + text[end_idx:]
                    text_lower = text.lower()
        return text
    
    def is_proper_noun(self, word: str, context: List[str]) -> bool:
        if len(word) > 1 and word[0].isupper() and word[1:].islower():
            return True
        if context and context[-1] in {'.', '!', '?'}:
            return True
        if word.lower() in self.proper_nouns:
            return True
        return False
    
    def correct_text(self, text: str) -> str:
        """Full correction pipeline with tense handling"""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        corrected_sentences = []
        
        for sentence in sentences:
            # Step 1: Correct common phrases
            sentence = self.correct_phrases(sentence)
            
            # Step 2: Correct tenses
            sentence = self.correct_tense(sentence)
            
            # Step 3: Word-level corrections
            tokens = re.findall(r"(\w+)|(\s+)|([^\w\s])", sentence)
            corrected = []
            
            for word, whitespace, punct in tokens:
                if word:
                    if self.is_proper_noun(word, corrected):
                        corrected.append(word)
                        continue
                        
                    suggestions = self._get_candidates(word)
                    if suggestions and suggestions[0].lower() != word.lower():
                        correction = suggestions[0]
                        if word[0].isupper() or (corrected and corrected[-1] in {'.', '!', '?'}):
                            correction = correction[0].upper() + correction[1:]
                        corrected.append(correction)
                    else:
                        corrected.append(word)
                if whitespace:
                    corrected.append(whitespace)
                if punct:
                    corrected.append(punct)
                    
            corrected_sentences.append(''.join(corrected))
            
        return ' '.join(corrected_sentences)

def main():
    st.set_page_config(page_title="AI Autocorrect Tool", page_icon="✍️")
    st.title("✍️ AI Autocorrect Tool")
    st.markdown("Improve text accuracy and fluency with AI-powered corrections")
    
    if 'autocorrect' not in st.session_state:
        st.session_state.autocorrect = AutocorrectTool()
        default_corpus = [
            "This is a sample text for building the autocorrect vocabulary.",
            "The tool will help correct spelling mistakes and improve writing.",
            "It uses edit distance and word frequency to make suggestions.",
            "Natural language processing techniques make it more effective.",
            "You can upload your own text files to customize the vocabulary.",
            "My name is Shreya and I live in Mumbai.",
            "Proper nouns like London and Microsoft should be preserved.",
            "Yesterday I was at the park. Today I am at home. Tomorrow I will go to school."
        ]
        st.session_state.autocorrect.build_vocabulary(default_corpus)
        st.session_state.autocorrect.proper_nouns.update(
            {'shreya', 'mumbai', 'london', 'microsoft'}
        )
    
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload a text file to enhance vocabulary", type=['txt'])
        if uploaded_file is not None:
            corpus = uploaded_file.read().decode("utf-8").splitlines()
            st.session_state.autocorrect.build_vocabulary(corpus)
            st.success(f"Vocabulary updated with {len(corpus)} lines")
        
        st.subheader("Autocorrect Options")
        max_distance = st.slider("Maximum edit distance", 1, 3, 2)
        st.session_state.autocorrect.max_edit_distance = max_distance
        top_suggestions = st.slider("Number of suggestions", 1, 10, 5)
        st.session_state.autocorrect.top_suggestions = top_suggestions
    
    tab1, tab2 = st.tabs(["Text Correction", "Examples"])
    
    with tab1:
        input_text = st.text_area("Enter your text here:", height=200,
                                placeholder="Type or paste your text to get autocorrect suggestions...")
        
        if st.button("Correct Text"):
            if input_text.strip():
                with st.spinner("Processing..."):
                    corrected_text = st.session_state.autocorrect.correct_text(input_text)
                
                st.subheader("Corrected Text")
                st.write(corrected_text)
                
                st.subheader("Changes Made")
                original_words = re.findall(r'\w+', input_text.lower())
                corrected_words = re.findall(r'\w+', corrected_text.lower())
                changes = [f"'{orig}'→'{corr}'" for orig, corr in zip(original_words, corrected_words) if orig != corr]
                st.write(", ".join(changes) if changes else "No corrections needed")
            else:
                st.warning("Please enter some text to correct")
    
    with tab2:
        st.subheader("Try These Examples:")
        examples = [
            "i name was shreya",
            "yesterday i is at school",
            "tomorrow i go to the park",
            "she were happy yesterday",
            "we has a meeting last week"
        ]
        for example in examples:
            if st.button(example):
                corrected = st.session_state.autocorrect.correct_text(example)
                st.write(f"**Original:** {example}")
                st.write(f"**Corrected:** {corrected}")

if __name__ == "__main__":
    main()