#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace text_utils {

// --- Basic trimming and normalization utilities ---------------------------------------------

std::string trim(const std::string &text) {
    auto first = std::find_if_not(text.begin(), text.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });

    if (first == text.end()) {
        return {};
    }

    auto last = std::find_if_not(text.rbegin(), text.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }).base();

    return std::string(first, last);
}

std::string toLower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

std::string toUpper(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });
    return text;
}

std::string normalizeForComparison(const std::string &text) {
    std::string normalized;
    normalized.reserve(text.size());

    for (unsigned char ch : text) {
        if (std::isalnum(ch) != 0) {
            normalized.push_back(static_cast<char>(std::tolower(ch)));
        }
    }

    return normalized;
}

bool equalsIgnoreCase(const std::string &lhs, const std::string &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(lhs[i])) !=
            std::tolower(static_cast<unsigned char>(rhs[i]))) {
            return false;
        }
    }
    return true;
}

// --- Tokenization helpers --------------------------------------------------------------------

bool isSentenceTerminator(char ch) {
    return ch == '.' || ch == '!' || ch == '?' || ch == ';';
}

std::vector<std::string> splitSentences(const std::string &text) {
    std::vector<std::string> sentences;
    std::string current;

    for (char ch : text) {
        current.push_back(ch);
        if (isSentenceTerminator(ch)) {
            const std::string trimmed = trim(current);
            if (!trimmed.empty()) {
                sentences.push_back(trimmed);
            }
            current.clear();
        }
    }

    const std::string trailing = trim(current);
    if (!trailing.empty()) {
        sentences.push_back(trailing);
    }

    return sentences;
}

std::vector<std::string> splitWords(const std::string &text) {
    std::vector<std::string> words;
    std::string current;

    auto flushCurrent = [&]() {
        if (!current.empty()) {
            words.push_back(current);
            current.clear();
        }
    };

    for (unsigned char ch : text) {
        if (std::isalnum(ch) != 0 || ch == '\'') {
            current.push_back(static_cast<char>(std::tolower(ch)));
        } else {
            flushCurrent();
        }
    }

    flushCurrent();
    return words;
}

std::vector<std::string> splitUniqueWords(const std::string &text) {
    std::vector<std::string> words = splitWords(text);
    std::sort(words.begin(), words.end());
    words.erase(std::unique(words.begin(), words.end()), words.end());
    return words;
}

// --- N-gram utilities -----------------------------------------------------------------------

using NGram = std::vector<std::string>;

struct NGramHash {
    std::size_t operator()(const NGram &ngram) const noexcept {
        std::size_t seed = ngram.size();
        for (const auto &token : ngram) {
            for (char ch : token) {
                seed ^= static_cast<std::size_t>(ch) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
            }
        }
        return seed;
    }
};

struct NGramEqual {
    bool operator()(const NGram &lhs, const NGram &rhs) const noexcept {
        return lhs == rhs;
    }
};

using NGramFrequencyMap = std::unordered_map<NGram, int, NGramHash, NGramEqual>;

NGramFrequencyMap buildNGrams(const std::vector<std::string> &words, int n) {
    NGramFrequencyMap frequencies;
    if (n <= 0 || static_cast<std::size_t>(n) > words.size()) {
        return frequencies;
    }

    for (std::size_t i = 0; i + static_cast<std::size_t>(n) <= words.size(); ++i) {
        NGram ngram;
        ngram.reserve(static_cast<std::size_t>(n));
        for (int offset = 0; offset < n; ++offset) {
            ngram.push_back(words[i + static_cast<std::size_t>(offset)]);
        }
        frequencies[ngram]++;
    }

    return frequencies;
}

std::string joinNGram(const NGram &ngram) {
    std::ostringstream stream;
    for (std::size_t i = 0; i < ngram.size(); ++i) {
        if (i > 0) {
            stream << " ";
        }
        stream << ngram[i];
    }
    return stream.str();
}

// --- Syllable and readability estimation -----------------------------------------------------

int estimateSyllables(const std::string &word) {
    if (word.empty()) {
        return 0;
    }

    int syllables = 0;
    bool previousVowel = false;

    auto isVowel = [](char ch) {
        char lower = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        return lower == 'a' || lower == 'e' || lower == 'i' || lower == 'o' || lower == 'u' || lower == 'y';
    };

    for (char ch : word) {
        if (isVowel(ch)) {
            if (!previousVowel) {
                ++syllables;
            }
            previousVowel = true;
        } else {
            previousVowel = false;
        }
    }

    if (!word.empty()) {
        char last = static_cast<char>(std::tolower(static_cast<unsigned char>(word.back())));
        if (last == 'e' && syllables > 1) {
            --syllables;
        }
    }

    if (syllables == 0) {
        syllables = 1;
    }

    return syllables;
}

double fleschReadingEase(int totalWords, int totalSentences, int totalSyllables) {
    if (totalWords == 0 || totalSentences == 0) {
        return 0.0;
    }

    const double wordsPerSentence = static_cast<double>(totalWords) / static_cast<double>(totalSentences);
    const double syllablesPerWord = static_cast<double>(totalSyllables) / static_cast<double>(totalWords);
    const double score = 206.835 - (1.015 * wordsPerSentence) - (84.6 * syllablesPerWord);
    return std::clamp(score, 0.0, 120.0);
}

double fleschKincaidGradeLevel(int totalWords, int totalSentences, int totalSyllables) {
    if (totalWords == 0 || totalSentences == 0) {
        return 0.0;
    }

    const double wordsPerSentence = static_cast<double>(totalWords) / static_cast<double>(totalSentences);
    const double syllablesPerWord = static_cast<double>(totalSyllables) / static_cast<double>(totalWords);
    return (0.39 * wordsPerSentence) + (11.8 * syllablesPerWord) - 15.59;
}

// --- Sentiment estimation --------------------------------------------------------------------

struct SentimentLexicon {
    std::unordered_set<std::string> positiveWords;
    std::unordered_set<std::string> negativeWords;

    SentimentLexicon() {
        positiveWords = {
            "amazing", "awesome", "beautiful", "best", "brilliant", "cheerful", "creative", "delightful",
            "elegant", "fantastic", "glorious", "happy", "impressive", "joy", "kind", "lucky", "marvelous",
            "nice", "optimistic", "positive", "smart", "terrific", "vibrant", "wonderful"};

        negativeWords = {
            "awful", "bad", "boring", "broken", "cruel", "damaged", "dark", "depressing", "evil", "fail",
            "gloomy", "horrible", "hurt", "imperfect", "jealous", "lonely", "mad", "negative", "pain",
            "sad", "terrible", "ugly", "worst", "worthless"};
    }
};

struct SentimentScore {
    int positiveMatches{};
    int negativeMatches{};
    double normalizedScore{}; // Range [-1.0, 1.0]
    std::string overallFeeling;
};

SentimentScore analyzeSentiment(const std::vector<std::string> &words, const SentimentLexicon &lexicon) {
    SentimentScore score;

    for (const auto &word : words) {
        if (lexicon.positiveWords.count(word) != 0) {
            ++score.positiveMatches;
        }
        if (lexicon.negativeWords.count(word) != 0) {
            ++score.negativeMatches;
        }
    }

    const int totalMatches = score.positiveMatches + score.negativeMatches;
    if (totalMatches > 0) {
        score.normalizedScore = static_cast<double>(score.positiveMatches - score.negativeMatches) /
                                static_cast<double>(totalMatches);
    } else {
        score.normalizedScore = 0.0;
    }

    if (score.normalizedScore > 0.3) {
        score.overallFeeling = "Positive";
    } else if (score.normalizedScore < -0.3) {
        score.overallFeeling = "Negative";
    } else if (totalMatches > 0) {
        score.overallFeeling = "Mixed";
    } else {
        score.overallFeeling = "Neutral";
    }

    return score;
}

// --- Keyword insight and thematic extraction -------------------------------------------------

struct KeywordInsight {
    std::string keyword;
    int frequency{};
    double percentageOfTotal{};
};

std::vector<KeywordInsight> extractTopKeywords(const std::vector<std::string> &words, std::size_t topN) {
    std::unordered_map<std::string, int> frequency;
    for (const auto &word : words) {
        if (!word.empty()) {
            ++frequency[word];
        }
    }

    std::vector<KeywordInsight> insights;
    insights.reserve(frequency.size());

    for (const auto &[keyword, count] : frequency) {
        KeywordInsight insight;
        insight.keyword = keyword;
        insight.frequency = count;
        insight.percentageOfTotal = words.empty() ? 0.0
                                                  : (static_cast<double>(count) * 100.0) /
                                                        static_cast<double>(words.size());
        insights.push_back(insight);
    }

    std::sort(insights.begin(), insights.end(), [](const KeywordInsight &lhs, const KeywordInsight &rhs) {
        if (lhs.frequency == rhs.frequency) {
            return lhs.keyword < rhs.keyword;
        }
        return lhs.frequency > rhs.frequency;
    });

    if (insights.size() > topN) {
        insights.resize(topN);
    }

    return insights;
}

// --- Character signature ---------------------------------------------------------------------

struct CharacterSignature {
    std::map<char, int> frequency;
    std::set<char> uniqueCharacters;
    char mostCommonCharacter{'\0'};
    int mostCommonCount{};
};

CharacterSignature buildCharacterSignature(const std::string &text) {
    CharacterSignature signature;

    for (unsigned char ch : text) {
        if (std::isprint(ch) == 0) {
            continue;
        }
        signature.frequency[static_cast<char>(ch)]++;
        signature.uniqueCharacters.insert(static_cast<char>(ch));
    }

    for (const auto &[character, count] : signature.frequency) {
        if (count > signature.mostCommonCount) {
            signature.mostCommonCount = count;
            signature.mostCommonCharacter = character;
        }
    }

    return signature;
}

// --- Sentence diagnostics --------------------------------------------------------------------

struct SentenceInsight {
    std::string text;
    int wordCount{};
    int syllableCount{};
    double averageWordLength{};
};

std::vector<SentenceInsight> analyzeSentences(const std::vector<std::string> &sentences) {
    std::vector<SentenceInsight> insights;
    insights.reserve(sentences.size());

    for (const auto &sentence : sentences) {
        SentenceInsight info;
        info.text = sentence;

        std::vector<std::string> words = splitWords(sentence);
        info.wordCount = static_cast<int>(words.size());
        if (!words.empty()) {
            int totalLetters = 0;
            for (const auto &word : words) {
                totalLetters += static_cast<int>(word.size());
                info.syllableCount += estimateSyllables(word);
            }
            info.averageWordLength = static_cast<double>(totalLetters) / static_cast<double>(words.size());
        }

        insights.push_back(info);
    }

    return insights;
}

// --- Text history ----------------------------------------------------------------------------

struct TextHistoryEntry {
    std::string originalInput;
    std::chrono::system_clock::time_point timestamp;
};

class TextHistory {
  public:
    void addEntry(const std::string &input) {
        entries.push_back({input, std::chrono::system_clock::now()});
        if (entries.size() > maximumEntries) {
            entries.erase(entries.begin());
        }
    }

    void printHistory() const {
        if (entries.empty()) {
            std::cout << "No history captured yet." << std::endl;
            return;
        }

        std::cout << "\n📚 Input History (most recent first)" << std::endl;
        std::cout << "-------------------------------------" << std::endl;

        for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
            const auto time = std::chrono::system_clock::to_time_t(it->timestamp);
            std::cout << "- [" << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S")
                      << "] \"" << it->originalInput << "\"" << std::endl;
        }
    }

  private:
    std::vector<TextHistoryEntry> entries;
    static constexpr std::size_t maximumEntries = 12;
};

// --- Text intelligence summary ---------------------------------------------------------------

struct TextAnalysis {
    std::string originalText;
    std::string trimmedText;
    std::string uppercaseText;
    std::string lowercaseText;
    std::vector<std::string> words;
    std::vector<std::string> uniqueWords;
    std::vector<std::string> sentences;
    std::vector<SentenceInsight> sentenceInsights;
    NGramFrequencyMap bigrams;
    NGramFrequencyMap trigrams;
    CharacterSignature characterSignature;
    SentimentScore sentiment;
    std::vector<KeywordInsight> keywords;
    SentenceInsight longestSentence;
    SentenceInsight shortestSentence;
    std::string longestWord;
    std::string shortestWord;
    bool palindrome{};

    int charCountWithSpaces{};
    int charCountWithoutSpaces{};
    int totalSyllables{};
    int totalWordLength{};

    double averageWordLength{};
    double averageSentenceLength{};
    double fleschScore{};
    double gradeLevel{};
};

// --- Core analysis engine --------------------------------------------------------------------

class TextIntelligenceEngine {
  public:
    TextIntelligenceEngine() : sentimentLexicon() {}

    TextAnalysis analyze(const std::string &text) {
        TextAnalysis analysis;
        analysis.originalText = text;
        analysis.trimmedText = trim(text);

        analysis.charCountWithSpaces = static_cast<int>(text.size());
        analysis.charCountWithoutSpaces = static_cast<int>(std::count_if(
            text.begin(), text.end(), [](unsigned char ch) {
                return std::isspace(ch) == 0;
            }));

        analysis.uppercaseText = toUpper(text);
        analysis.lowercaseText = toLower(text);
        analysis.words = splitWords(text);
        analysis.uniqueWords = splitUniqueWords(text);
        analysis.sentences = splitSentences(text);

        analysis.sentenceInsights = analyzeSentences(analysis.sentences);
        analysis.bigrams = buildNGrams(analysis.words, 2);
        analysis.trigrams = buildNGrams(analysis.words, 3);
        analysis.characterSignature = buildCharacterSignature(text);
        analysis.sentiment = analyzeSentiment(analysis.words, sentimentLexicon);
        analysis.keywords = extractTopKeywords(analysis.words, 8);

        processWordStatistics(analysis);
        processSentenceStatistics(analysis);
        processPalindrome(analysis);
        processReadability(analysis);

        return analysis;
    }

    void printAnalysis(const TextAnalysis &analysis) const {
        printHeader();
        printOriginalTextSection(analysis);
        printMeasurementsSection(analysis);
        printSentenceDiagnostics(analysis);
        printKeywordHighlights(analysis);
        printNGramHighlights(analysis);
        printCharacterSignature(analysis);
        printSentiment(analysis);
        printRecommendations(analysis);
    }

  private:
    SentimentLexicon sentimentLexicon;

    static void processWordStatistics(TextAnalysis &analysis) {
        analysis.totalSyllables = 0;
        analysis.totalWordLength = 0;
        for (const auto &word : analysis.words) {
            analysis.totalSyllables += estimateSyllables(word);
            analysis.totalWordLength += static_cast<int>(word.size());
            if (word.size() > analysis.longestWord.size()) {
                analysis.longestWord = word;
            }
            if (analysis.shortestWord.empty() || word.size() < analysis.shortestWord.size()) {
                analysis.shortestWord = word;
            }
        }

        if (!analysis.words.empty()) {
            analysis.averageWordLength = static_cast<double>(analysis.totalWordLength) /
                                         static_cast<double>(analysis.words.size());
        }
    }

    static void processSentenceStatistics(TextAnalysis &analysis) {
        if (!analysis.sentenceInsights.empty()) {
            analysis.longestSentence = *std::max_element(
                analysis.sentenceInsights.begin(), analysis.sentenceInsights.end(),
                [](const SentenceInsight &lhs, const SentenceInsight &rhs) {
                    return lhs.wordCount < rhs.wordCount;
                });

            analysis.shortestSentence = *std::min_element(
                analysis.sentenceInsights.begin(), analysis.sentenceInsights.end(),
                [](const SentenceInsight &lhs, const SentenceInsight &rhs) {
                    return lhs.wordCount < rhs.wordCount;
                });

            const double totalWordsAcrossSentences = std::accumulate(
                analysis.sentenceInsights.begin(), analysis.sentenceInsights.end(), 0.0,
                [](double sum, const SentenceInsight &info) {
                    return sum + static_cast<double>(info.wordCount);
                });

            analysis.averageSentenceLength = totalWordsAcrossSentences /
                                             static_cast<double>(analysis.sentenceInsights.size());
        }
    }

    static void processPalindrome(TextAnalysis &analysis) {
        const std::string normalized = normalizeForComparison(analysis.trimmedText);
        std::string reversed = normalized;
        std::reverse(reversed.begin(), reversed.end());
        analysis.palindrome = !normalized.empty() && normalized == reversed;
    }

    static void processReadability(TextAnalysis &analysis) {
        const int totalWords = static_cast<int>(analysis.words.size());
        const int totalSentences = static_cast<int>(analysis.sentences.size());
        analysis.fleschScore = fleschReadingEase(totalWords, totalSentences, analysis.totalSyllables);
        analysis.gradeLevel = fleschKincaidGradeLevel(totalWords, totalSentences, analysis.totalSyllables);
    }

    static void printHeader() {
        std::cout << "\n🧠 Ultra Text Intelligence Report" << std::endl;
        std::cout << "==================================" << std::endl;
    }

    static void printOriginalTextSection(const TextAnalysis &analysis) {
        std::cout << "\n📝 Core Text Views" << std::endl;
        std::cout << "------------------" << std::endl;
        std::cout << "Original: " << analysis.originalText << std::endl;
        std::cout << "Trimmed: " << analysis.trimmedText << std::endl;
        std::cout << "Uppercase: " << analysis.uppercaseText << std::endl;
        std::cout << "Lowercase: " << analysis.lowercaseText << std::endl;
    }

    static void printMeasurementsSection(const TextAnalysis &analysis) {
        std::cout << "\n📊 Measurements" << std::endl;
        std::cout << "----------------" << std::endl;
        std::cout << "- Characters (with spaces): " << analysis.charCountWithSpaces << std::endl;
        std::cout << "- Characters (without spaces): " << analysis.charCountWithoutSpaces << std::endl;
        std::cout << "- Word count: " << analysis.words.size() << std::endl;
        std::cout << "- Unique word count: " << analysis.uniqueWords.size() << std::endl;
        std::cout << "- Average word length: " << std::fixed << std::setprecision(2)
                  << analysis.averageWordLength << std::endl;
        std::cout << "- Longest word: "
                  << (analysis.longestWord.empty() ? "<none>" : analysis.longestWord) << std::endl;
        std::cout << "- Shortest word: "
                  << (analysis.shortestWord.empty() ? "<none>" : analysis.shortestWord) << std::endl;
        std::cout << "- Sentence count: " << analysis.sentences.size() << std::endl;
        std::cout << "- Average sentence length: " << std::fixed << std::setprecision(2)
                  << analysis.averageSentenceLength << " words" << std::endl;
        std::cout << "- Total syllables (estimated): " << analysis.totalSyllables << std::endl;
        std::cout << "- Flesch Reading Ease: " << std::fixed << std::setprecision(2)
                  << analysis.fleschScore << std::endl;
        std::cout << "- Flesch-Kincaid Grade Level: " << std::fixed << std::setprecision(2)
                  << analysis.gradeLevel << std::endl;
        std::cout << "- Palindrome (ignoring punctuation & case): "
                  << (analysis.palindrome ? "Yes" : "No") << std::endl;
    }

    static void printSentenceDiagnostics(const TextAnalysis &analysis) {
        std::cout << "\n📐 Sentence Diagnostics" << std::endl;
        std::cout << "------------------------" << std::endl;

        if (analysis.sentenceInsights.empty()) {
            std::cout << "No sentence boundaries detected." << std::endl;
            return;
        }

        std::cout << "Longest sentence (" << analysis.longestSentence.wordCount << " words):\n"
                  << "  " << analysis.longestSentence.text << std::endl;

        std::cout << "Shortest sentence (" << analysis.shortestSentence.wordCount << " words):\n"
                  << "  " << analysis.shortestSentence.text << std::endl;

        std::cout << "\nDetailed breakdown:" << std::endl;
        int index = 1;
        for (const auto &info : analysis.sentenceInsights) {
            std::cout << "  [" << index++ << "] " << info.wordCount << " words, " << info.syllableCount
                      << " syllables (avg word length: " << std::fixed << std::setprecision(2)
                      << info.averageWordLength << ")\n"
                      << "      " << info.text << std::endl;
        }
    }

    static void printKeywordHighlights(const TextAnalysis &analysis) {
        std::cout << "\n🔑 Keyword Highlights" << std::endl;
        std::cout << "---------------------" << std::endl;

        if (analysis.keywords.empty()) {
            std::cout << "Not enough content to extract keywords." << std::endl;
            return;
        }

        for (const auto &keyword : analysis.keywords) {
            std::cout << "- " << keyword.keyword << ": " << keyword.frequency << " occurrences ("
                      << std::fixed << std::setprecision(2) << keyword.percentageOfTotal << "% of text)"
                      << std::endl;
        }
    }

    static void printNGramHighlights(const TextAnalysis &analysis) {
        std::cout << "\n🧩 N-gram Highlights" << std::endl;
        std::cout << "--------------------" << std::endl;
        printTopNGrams(analysis.bigrams, "Bigrams");
        printTopNGrams(analysis.trigrams, "Trigrams");
    }

    static void printTopNGrams(const NGramFrequencyMap &ngrams, const std::string &title) {
        std::cout << title << ":" << std::endl;
        if (ngrams.empty()) {
            std::cout << "  <none>" << std::endl;
            return;
        }

        std::vector<std::pair<NGram, int>> sorted(ngrams.begin(), ngrams.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.second == rhs.second) {
                return lhs.first < rhs.first;
            }
            return lhs.second > rhs.second;
        });

        const std::size_t limit = std::min<std::size_t>(sorted.size(), 5);
        for (std::size_t i = 0; i < limit; ++i) {
            std::cout << "  - " << joinNGram(sorted[i].first) << " (" << sorted[i].second << " occurrences)"
                      << std::endl;
        }
    }

    static void printCharacterSignature(const TextAnalysis &analysis) {
        std::cout << "\n🔠 Character Signature" << std::endl;
        std::cout << "----------------------" << std::endl;
        const auto &signature = analysis.characterSignature;
        std::cout << "Unique printable characters: " << signature.uniqueCharacters.size() << std::endl;
        if (signature.mostCommonCount > 0) {
            std::cout << "Most common character: '" << signature.mostCommonCharacter << "' ("
                      << signature.mostCommonCount << " times)" << std::endl;
        }

        std::cout << "Top characters:" << std::endl;
        if (signature.frequency.empty()) {
            std::cout << "  <none>" << std::endl;
            return;
        }

        std::vector<std::pair<char, int>> sorted(signature.frequency.begin(), signature.frequency.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.second == rhs.second) {
                return lhs.first < rhs.first;
            }
            return lhs.second > rhs.second;
        });

        const std::size_t limit = std::min<std::size_t>(sorted.size(), 10);
        for (std::size_t i = 0; i < limit; ++i) {
            const double relative = static_cast<double>(sorted[i].second) * 100.0 /
                                    static_cast<double>(analysis.charCountWithSpaces);
            std::cout << "  - '" << sorted[i].first << "' -> " << sorted[i].second << " times ("
                      << std::fixed << std::setprecision(2) << relative << "%)" << std::endl;
        }
    }

    static void printSentiment(const TextAnalysis &analysis) {
        std::cout << "\n💬 Sentiment Snapshot" << std::endl;
        std::cout << "---------------------" << std::endl;
        const auto &sentiment = analysis.sentiment;
        std::cout << "- Positive matches: " << sentiment.positiveMatches << std::endl;
        std::cout << "- Negative matches: " << sentiment.negativeMatches << std::endl;
        std::cout << "- Sentiment score: " << std::fixed << std::setprecision(2) << sentiment.normalizedScore
                  << std::endl;
        std::cout << "- Overall feeling: " << sentiment.overallFeeling << std::endl;

        if (sentiment.overallFeeling == "Neutral") {
            std::cout << "Tip: Try adding emotional adjectives to shift the sentiment." << std::endl;
        } else if (sentiment.overallFeeling == "Mixed") {
            std::cout << "Tip: Your text has balanced emotion. Emphasize one side for clarity." << std::endl;
        } else if (sentiment.overallFeeling == "Negative") {
            std::cout << "Tip: Consider injecting positive language to brighten the tone." << std::endl;
        } else {
            std::cout << "Tip: Positive tone detected. Keep the momentum going!" << std::endl;
        }
    }

    static void printRecommendations(const TextAnalysis &analysis) {
        std::cout << "\n✨ Intelligent Recommendations" << std::endl;
        std::cout << "-----------------------------" << std::endl;

        if (analysis.words.size() > 50 && analysis.gradeLevel > 12.0) {
            std::cout << "- Your text reads at a college level. Simplify sentences to reach a broader audience."
                      << std::endl;
        } else if (analysis.gradeLevel < 6.0) {
            std::cout << "- The language is very easy to digest. Add descriptive words for richer storytelling."
                      << std::endl;
        } else {
            std::cout << "- Complexity is moderate. Maintain sentence variety to keep readers engaged."
                      << std::endl;
        }

        if (analysis.sentiment.overallFeeling == "Negative") {
            std::cout << "- Consider rebalancing with positive phrasing to lift the mood." << std::endl;
        }

        if (!analysis.palindrome && analysis.words.size() >= 3) {
            std::cout << "- Try crafting a palindrome challenge using your theme. It's fun and mind-bending!"
                      << std::endl;
        }

        if (!analysis.uniqueWords.empty()) {
            std::cout << "- You used " << analysis.uniqueWords.size() << " unique words. Explore synonyms to expand"
                      << " the vocabulary range." << std::endl;
        }
    }
};

} // namespace text_utils

// --- Interactive console application ---------------------------------------------------------

class ConsoleUI {
  public:
    ConsoleUI() : engine(), history() {}

    void run() {
        printWelcome();
        std::string userInput;

        while (true) {
            const int choice = promptMenu();
            if (choice == 5) {
                std::cout << "Thanks for exploring the Ultra Text Intelligence Console. 👋" << std::endl;
                break;
            }

            switch (choice) {
            case 1:
                analyzeSingleEntry();
                break;
            case 2:
                analyzeBatchEntries();
                break;
            case 3:
                history.printHistory();
                break;
            case 4:
                printHelp();
                break;
            default:
                std::cout << "Unknown option selected." << std::endl;
                break;
            }
        }
    }

  private:
    text_utils::TextIntelligenceEngine engine;
    text_utils::TextHistory history;

    static void printWelcome() {
        std::cout << "✨ Welcome to the Ultra Text Intelligence Console ✨" << std::endl;
        std::cout << "Discover deep insights, readability metrics, sentiment, and more." << std::endl;
    }

    static void printHelp() {
        std::cout << "\nℹ️ Help & Tips" << std::endl;
        std::cout << "--------------" << std::endl;
        std::cout << "1. Analyze a single entry: Paste or type text and receive an extensive report." << std::endl;
        std::cout << "2. Batch analysis: Evaluate multiple lines, perfect for brainstorming sessions." << std::endl;
        std::cout << "3. View history: Review the most recent 12 inputs processed." << std::endl;
        std::cout << "4. Help menu: You're here right now!" << std::endl;
        std::cout << "5. Exit: Close the console experience." << std::endl;
    }

    static int promptMenu() {
        std::cout << "\nMain Menu" << std::endl;
        std::cout << "---------" << std::endl;
        std::cout << "1. Analyze single entry" << std::endl;
        std::cout << "2. Analyze batch entries" << std::endl;
        std::cout << "3. View input history" << std::endl;
        std::cout << "4. Help & tips" << std::endl;
        std::cout << "5. Exit" << std::endl;
        std::cout << "Select an option (1-5): ";

        int choice = 0;
        while (!(std::cin >> choice) || choice < 1 || choice > 5) {
            std::cout << "Invalid choice. Please enter a number between 1 and 5: ";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        return choice;
    }

    void analyzeSingleEntry() {
        std::cout << "\nPlease enter your text (single paragraph)." << std::endl;
        std::cout << "Finish input with an empty line." << std::endl;

        std::string input = captureMultilineInput();
        if (input.empty()) {
            std::cout << "No text captured. Returning to menu." << std::endl;
            return;
        }

        history.addEntry(input);
        const auto analysis = engine.analyze(input);
        engine.printAnalysis(analysis);
    }

    void analyzeBatchEntries() {
        std::cout << "\nBatch mode activated." << std::endl;
        std::cout << "Enter multiple lines. Type a single line with 'END' to finish." << std::endl;

        std::vector<std::string> lines;
        std::string line;
        while (true) {
            std::getline(std::cin, line);
            if (text_utils::equalsIgnoreCase(text_utils::trim(line), "END")) {
                break;
            }
            lines.push_back(line);
        }

        if (lines.empty()) {
            std::cout << "No lines collected. Returning to menu." << std::endl;
            return;
        }

        std::ostringstream combined;
        for (const auto &entry : lines) {
            combined << entry << '\n';
        }

        std::string text = combined.str();
        history.addEntry(text_utils::trim(text));
        const auto analysis = engine.analyze(text);
        engine.printAnalysis(analysis);

        printBatchSummary(lines);
    }

    static std::string captureMultilineInput() {
        std::ostringstream buffer;
        std::string line;
        while (true) {
            if (!std::getline(std::cin, line)) {
                break;
            }
            if (line.empty()) {
                break;
            }
            buffer << line << '\n';
        }
        return text_utils::trim(buffer.str());
    }

    static void printBatchSummary(const std::vector<std::string> &lines) {
        std::cout << "\n📦 Batch Summary" << std::endl;
        std::cout << "----------------" << std::endl;
        std::cout << "- Total lines processed: " << lines.size() << std::endl;
        std::size_t maxLength = 0;
        std::size_t minLength = std::numeric_limits<std::size_t>::max();

        for (const auto &line : lines) {
            maxLength = std::max(maxLength, line.size());
            minLength = std::min(minLength, line.size());
        }

        if (!lines.empty()) {
            const double averageLength = std::accumulate(
                lines.begin(), lines.end(), 0.0,
                [](double sum, const std::string &entry) {
                    return sum + static_cast<double>(entry.size());
                }) /
                                        static_cast<double>(lines.size());
            std::cout << "- Longest line length: " << maxLength << std::endl;
            std::cout << "- Shortest line length: " << minLength << std::endl;
            std::cout << "- Average line length: " << std::fixed << std::setprecision(2) << averageLength
                      << std::endl;
        }

        std::cout << "- Preview of first three lines:" << std::endl;
        const std::size_t previewCount = std::min<std::size_t>(3, lines.size());
        for (std::size_t i = 0; i < previewCount; ++i) {
            std::cout << "  " << (i + 1) << ": " << lines[i] << std::endl;
        }
    }
};

int main() {
    ConsoleUI ui;
    ui.run();
    return 0;
}