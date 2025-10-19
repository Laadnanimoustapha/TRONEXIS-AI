#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <regex> // Added for enhanced pattern matching

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

std::string toTitleCase(const std::string &text) {
    std::string result;
    result.reserve(text.size());
    bool capitalizeNext = true;

    for (unsigned char ch : text) {
        if (std::isspace(ch) != 0) {
            capitalizeNext = true;
            result.push_back(static_cast<char>(ch));
            continue;
        }

        if (capitalizeNext) {
            result.push_back(static_cast<char>(std::toupper(ch)));
            capitalizeNext = false;
        } else {
            result.push_back(static_cast<char>(std::tolower(ch)));
        }
    }

    return result;
}

std::string reverseText(const std::string &text) {
    std::string reversed = text;
    std::reverse(reversed.begin(), reversed.end());
    return reversed;
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

// --- Enhanced tokenization helpers ----------------------------------------------------------

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

std::vector<std::string> extractUppercaseWords(const std::string &text) {
    std::vector<std::string> uppercaseWords;
    std::string current;

    auto flushWord = [&]() {
        if (current.size() > 1 && std::all_of(current.begin(), current.end(), [](unsigned char ch) {
                                    return std::isupper(ch) != 0;
                                })) {
            uppercaseWords.push_back(current);
        }
        current.clear();
    };

    for (unsigned char ch : text) {
        if (std::isalpha(ch) != 0) {
            current.push_back(static_cast<char>(ch));
        } else {
            flushWord();
        }
    }

    flushWord();
    return uppercaseWords;
}

// --- NEW: Enhanced pattern extraction -------------------------------------------------------

std::vector<std::string> extractEmails(const std::string &text) {
    std::vector<std::string> emails;
    std::regex email_pattern(R"(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)");
    
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), email_pattern);
    auto words_end = std::sregex_iterator();
    
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        emails.push_back((*i).str());
    }
    
    return emails;
}

std::vector<std::string> extractUrls(const std::string &text) {
    std::vector<std::string> urls;
    std::regex url_pattern(R"((https?://|www\.)[^\s]+)");
    
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), url_pattern);
    auto words_end = std::sregex_iterator();
    
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        urls.push_back((*i).str());
    }
    
    return urls;
}

std::vector<std::string> extractHashtags(const std::string &text) {
    std::vector<std::string> hashtags;
    std::regex hashtag_pattern(R"(#\w+)");
    
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), hashtag_pattern);
    auto words_end = std::sregex_iterator();
    
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        hashtags.push_back((*i).str());
    }
    
    return hashtags;
}

// --- NEW: Advanced text statistics ----------------------------------------------------------

struct AdvancedStats {
    int paragraphCount{};
    int questionCount{};
    int exclamationCount{};
    int quoteCount{};
    int parenthesisCount{};
    int dashCount{};
    double wordLengthVariance{};
    double sentenceLengthVariance{};
    std::string mostFrequentBigram;
    std::string mostFrequentTrigram;
};

AdvancedStats computeAdvancedStats(const std::string &text, const std::vector<std::string> &words, 
                                   const std::vector<std::string> &sentences) {
    AdvancedStats stats;
    
    // Count special characters
    for (char ch : text) {
        if (ch == '?') stats.questionCount++;
        else if (ch == '!') stats.exclamationCount++;
        else if (ch == '\"' || ch == '\'') stats.quoteCount++;
        else if (ch == '(' || ch == ')') stats.parenthesisCount++;
        else if (ch == '-' || ch == '–') stats.dashCount++;
    }
    
    // Count paragraphs (lines with content)
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) {
        if (!trim(line).empty()) {
            stats.paragraphCount++;
        }
    }
    
    // Calculate word length variance
    if (!words.empty()) {
        double mean = 0.0;
        for (const auto& word : words) {
            mean += word.length();
        }
        mean /= words.size();
        
        double variance = 0.0;
        for (const auto& word : words) {
            variance += std::pow(word.length() - mean, 2);
        }
        stats.wordLengthVariance = variance / words.size();
    }
    
    // Calculate sentence length variance
    if (!sentences.empty()) {
        double mean = 0.0;
        for (const auto& sentence : sentences) {
            mean += splitWords(sentence).size();
        }
        mean /= sentences.size();
        
        double variance = 0.0;
        for (const auto& sentence : sentences) {
            auto sentenceWords = splitWords(sentence);
            variance += std::pow(sentenceWords.size() - mean, 2);
        }
        stats.sentenceLengthVariance = variance / sentences.size();
    }
    
    return stats;
}

// --- Frequency helpers -----------------------------------------------------------------------

struct FrequencyEntry {
    std::string token;
    int count{};
    double percentage{};
};

std::vector<FrequencyEntry> computeAffixFrequency(const std::vector<std::string> &words,
                                                  std::size_t affixLength,
                                                  bool prefix,
                                                  std::size_t topN) {
    std::unordered_map<std::string, int> frequency;

    for (const auto &word : words) {
        if (word.size() < affixLength) {
            continue;
        }
        const std::string affix = prefix ? word.substr(0, affixLength)
                                         : word.substr(word.size() - affixLength, affixLength);
        frequency[affix]++;
    }

    std::vector<FrequencyEntry> results;
    results.reserve(frequency.size());
    for (const auto &[token, count] : frequency) {
        FrequencyEntry entry;
        entry.token = token;
        entry.count = count;
        entry.percentage = words.empty() ? 0.0
                                         : (static_cast<double>(count) * 100.0) / static_cast<double>(words.size());
        results.push_back(entry);
    }

    std::sort(results.begin(), results.end(), [](const FrequencyEntry &lhs, const FrequencyEntry &rhs) {
        if (lhs.count == rhs.count) {
            return lhs.token < rhs.token;
        }
        return lhs.count > rhs.count;
    });

    if (results.size() > topN) {
        results.resize(topN);
    }

    return results;
}

std::vector<FrequencyEntry> computeLetterPairs(const std::string &text, std::size_t topN) {
    std::unordered_map<std::string, int> frequency;
    std::string normalized = normalizeForComparison(text);

    if (normalized.size() < 2) {
        return {};
    }

    for (std::size_t i = 0; i + 1 < normalized.size(); ++i) {
        const std::string pair = normalized.substr(i, 2);
        frequency[pair]++;
    }

    std::vector<FrequencyEntry> results;
    results.reserve(frequency.size());

    for (const auto &[pair, count] : frequency) {
        FrequencyEntry entry;
        entry.token = pair;
        entry.count = count;
        entry.percentage = normalized.empty()
                                ? 0.0
                                : (static_cast<double>(count) * 100.0) / static_cast<double>(normalized.size());
        results.push_back(entry);
    }

    std::sort(results.begin(), results.end(), [](const FrequencyEntry &lhs, const FrequencyEntry &rhs) {
        if (lhs.count == rhs.count) {
            return lhs.token < rhs.token;
        }
        return lhs.count > rhs.count;
    });

    if (results.size() > topN) {
        results.resize(topN);
    }

    return results;
}

std::vector<std::string> detectRepeatedSentences(const std::vector<std::string> &sentences) {
    std::unordered_map<std::string, int> counts;
    for (const auto &sentence : sentences) {
        const std::string canonical = toLower(trim(sentence));
        counts[canonical]++;
    }

    std::vector<std::string> repeated;
    for (const auto &[sentence, count] : counts) {
        if (count > 1) {
            repeated.push_back(sentence);
        }
    }

    return repeated;
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

// --- NEW: Find most frequent n-grams --------------------------------------------------------

std::pair<std::string, int> findMostFrequentNGram(const NGramFrequencyMap &ngrams) {
    if (ngrams.empty()) {
        return {"", 0};
    }
    
    auto maxIt = std::max_element(ngrams.begin(), ngrams.end(),
        [](const auto &a, const auto &b) {
            return a.second < b.second;
        });
    
    return {joinNGram(maxIt->first), maxIt->second};
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

    // FIXED BUG: Handle words with no vowels (like "nth")
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

double gunningFogIndex(int totalWords, int totalSentences, int complexWords) {
    if (totalWords == 0 || totalSentences == 0) {
        return 0.0;
    }

    const double averageWordsPerSentence = static_cast<double>(totalWords) / static_cast<double>(totalSentences);
    const double complexWordPercentage = static_cast<double>(complexWords) * 100.0 / static_cast<double>(totalWords);
    return 0.4 * (averageWordsPerSentence + complexWordPercentage);
}

double smogIndex(int totalSentences, int complexWords) {
    if (totalSentences == 0 || complexWords == 0) {
        return 0.0;
    }

    return 1.0430 * std::sqrt(static_cast<double>(complexWords) * (30.0 / static_cast<double>(totalSentences))) +
           3.1291;
}

double colemanLiauIndex(int totalWords, int totalSentences, int totalLetters) {
    if (totalWords == 0 || totalSentences == 0) {
        return 0.0;
    }

    const double L = (static_cast<double>(totalLetters) / static_cast<double>(totalWords)) * 100.0;
    const double S = (static_cast<double>(totalSentences) / static_cast<double>(totalWords)) * 100.0;
    return (0.0588 * L) - (0.296 * S) - 15.8;
}

double automatedReadabilityIndex(int totalWords, int totalSentences, int totalLetters) {
    if (totalWords == 0 || totalSentences == 0) {
        return 0.0;
    }

    const double charactersPerWord = static_cast<double>(totalLetters) / static_cast<double>(totalWords);
    const double wordsPerSentence = static_cast<double>(totalWords) / static_cast<double>(totalSentences);
    return (4.71 * charactersPerWord) + (0.5 * wordsPerSentence) - 21.43;
}

// --- Sentiment estimation --------------------------------------------------------------------

class SentimentLexicon {
  public:
    SentimentLexicon() {
        positiveWords = {
            "amazing", "awesome", "beautiful", "best", "brilliant", "cheerful", "creative", "delightful",
            "elegant", "fantastic", "glorious", "happy", "impressive", "joy", "kind", "lucky", "marvelous",
            "nice", "optimistic", "positive", "smart", "terrific", "vibrant", "wonderful", "love", "excellent",
            "perfect", "great", "outstanding", "superb", "wonderful", "pleasure", "success", "win", "victory"};

        negativeWords = {
            "awful", "bad", "boring", "broken", "cruel", "damaged", "dark", "depressing", "evil", "fail",
            "gloomy", "horrible", "hurt", "imperfect", "jealous", "lonely", "mad", "negative", "pain",
            "sad", "terrible", "ugly", "worst", "worthless", "hate", "awful", "disappointing", "failure",
            "problem", "issue", "wrong", "mistake", "error"};
    }

    void addPositive(const std::string &word) {
        positiveWords.insert(toLower(word));
    }

    void addNegative(const std::string &word) {
        negativeWords.insert(toLower(word));
    }

    bool removePositive(const std::string &word) {
        return positiveWords.erase(toLower(word)) > 0;
    }

    bool removeNegative(const std::string &word) {
        return negativeWords.erase(toLower(word)) > 0;
    }

    void printSummary() const {
        std::cout << "\nLexicon summary" << std::endl;
        std::cout << "----------------" << std::endl;
        std::cout << "Positive words: " << positiveWords.size() << std::endl;
        std::cout << "Negative words: " << negativeWords.size() << std::endl;

        std::cout << "Sample positives: ";
        int showcase = 0;
        for (const auto &word : positiveWords) {
            std::cout << word << " ";
            if (++showcase >= 8) {
                break;
            }
        }
        if (positiveWords.empty()) {
            std::cout << "<empty>";
        }
        std::cout << std::endl;

        std::cout << "Sample negatives: ";
        showcase = 0;
        for (const auto &word : negativeWords) {
            std::cout << word << " ";
            if (++showcase >= 8) {
                break;
            }
        }
        if (negativeWords.empty()) {
            std::cout << "<empty>";
        }
        std::cout << std::endl;
    }

    const std::unordered_set<std::string> &positives() const {
        return positiveWords;
    }

    const std::unordered_set<std::string> &negatives() const {
        return negativeWords;
    }

    // NEW: Load from file
    bool loadFromFile(const std::string &filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;

            if (line[0] == '+') {
                addPositive(line.substr(1));
            } else if (line[0] == '-') {
                addNegative(line.substr(1));
            }
        }

        file.close();
        return true;
    }

    // NEW: Save to file
    bool saveToFile(const std::string &filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }

        file << "# Sentiment Lexicon\n";
        file << "# + for positive, - for negative\n";
        
        for (const auto &word : positiveWords) {
            file << "+" << word << "\n";
        }
        
        for (const auto &word : negativeWords) {
            file << "-" << word << "\n";
        }

        file.close();
        return true;
    }

  private:
    std::unordered_set<std::string> positiveWords;
    std::unordered_set<std::string> negativeWords;
};

struct SentimentScore {
    int positiveMatches{};
    int negativeMatches{};
    double normalizedScore{}; // Range [-1.0, 1.0]
    std::string overallFeeling;
    double confidence{}; // NEW: Confidence level
};

SentimentScore analyzeSentiment(const std::vector<std::string> &words, const SentimentLexicon &lexicon) {
    SentimentScore score;

    for (const auto &word : words) {
        if (lexicon.positives().count(word) != 0) {
            ++score.positiveMatches;
        }
        if (lexicon.negatives().count(word) != 0) {
            ++score.negativeMatches;
        }
    }

    const int totalMatches = score.positiveMatches + score.negativeMatches;
    if (totalMatches > 0) {
        score.normalizedScore = static_cast<double>(score.positiveMatches - score.negativeMatches) /
                                static_cast<double>(totalMatches);
        score.confidence = static_cast<double>(totalMatches) / words.size(); // NEW: Calculate confidence
    } else {
        score.normalizedScore = 0.0;
        score.confidence = 0.0;
    }

    // Enhanced sentiment classification with confidence
    if (std::abs(score.normalizedScore) < 0.1 || score.confidence < 0.1) {
        score.overallFeeling = "Neutral";
    } else if (score.normalizedScore > 0.3) {
        score.overallFeeling = "Positive";
    } else if (score.normalizedScore < -0.3) {
        score.overallFeeling = "Negative";
    } else {
        score.overallFeeling = "Mixed";
    }

    return score;
}

// --- Keyword insight and thematic extraction -------------------------------------------------

struct KeywordInsight {
    std::string keyword;
    int frequency{};
    double percentageOfTotal{};
    double tfidfScore{}; // NEW: TF-IDF score for better keyword extraction
};

// NEW: Calculate TF-IDF scores
std::vector<KeywordInsight> extractTopKeywordsWithTFIDF(const std::vector<std::string> &words, 
                                                       const std::vector<std::vector<std::string>> &allDocuments,
                                                       std::size_t topN) {
    std::unordered_map<std::string, int> frequency;
    std::unordered_map<std::string, int> documentFrequency;
    
    // Calculate term frequency in current document
    for (const auto &word : words) {
        if (!word.empty()) {
            ++frequency[word];
        }
    }
    
    // Calculate document frequency across all documents
    for (const auto &doc : allDocuments) {
        std::unordered_set<std::string> uniqueWords(doc.begin(), doc.end());
        for (const auto &word : uniqueWords) {
            if (frequency.count(word)) {
                documentFrequency[word]++;
            }
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
        
        // Calculate TF-IDF
        double tf = static_cast<double>(count) / words.size();
        double idf = std::log(static_cast<double>(allDocuments.size()) / 
                             (1.0 + documentFrequency[keyword]));
        insight.tfidfScore = tf * idf;
        
        insights.push_back(insight);
    }

    // Sort by TF-IDF score instead of just frequency
    std::sort(insights.begin(), insights.end(), [](const KeywordInsight &lhs, const KeywordInsight &rhs) {
        if (std::abs(lhs.tfidfScore - rhs.tfidfScore) < 0.001) {
            return lhs.frequency > rhs.frequency;
        }
        return lhs.tfidfScore > rhs.tfidfScore;
    });

    if (insights.size() > topN) {
        insights.resize(topN);
    }

    return insights;
}

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
    double entropy{}; // NEW: Information entropy of character distribution
};

// NEW: Calculate information entropy
double calculateEntropy(const std::map<char, int> &frequency,