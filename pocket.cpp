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
double calculateEntropy(const std::map<char, int> &frequency, int totalChars) {
    if (totalChars == 0) return 0.0;
    
    double entropy = 0.0;
    for (const auto &[ch, count] : frequency) {
        double probability = static_cast<double>(count) / totalChars;
        entropy -= probability * std::log2(probability);
    }
    return entropy;
}

CharacterSignature buildCharacterSignature(const std::string &text) {
    CharacterSignature signature;

    int totalChars = 0;
    for (unsigned char ch : text) {
        if (std::isprint(ch) == 0) {
            continue;
        }
        signature.frequency[static_cast<char>(ch)]++;
        signature.uniqueCharacters.insert(static_cast<char>(ch));
        totalChars++;
    }

    for (const auto &[character, count] : signature.frequency) {
        if (count > signature.mostCommonCount) {
            signature.mostCommonCount = count;
            signature.mostCommonCharacter = character;
        }
    }

    // NEW: Calculate entropy
    signature.entropy = calculateEntropy(signature.frequency, totalChars);

    return signature;
}

// --- Sentence diagnostics --------------------------------------------------------------------

struct SentenceInsight {
    std::string text;
    int wordCount{};
    int syllableCount{};
    double averageWordLength{};
    double complexityScore{}; // NEW: Sentence complexity score
};

// NEW: Calculate sentence complexity
double calculateSentenceComplexity(const std::vector<std::string> &words, int syllableCount) {
    if (words.empty()) return 0.0;
    
    double longWordRatio = 0.0;
    for (const auto &word : words) {
        if (word.length() > 6) {
            longWordRatio += 1.0;
        }
    }
    longWordRatio /= words.size();
    
    double avgSyllables = static_cast<double>(syllableCount) / words.size();
    
    return (longWordRatio + avgSyllables) / 2.0;
}

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
            info.complexityScore = calculateSentenceComplexity(words, info.syllableCount);
        }

        insights.push_back(info);
    }

    return insights;
}

// --- Patterns and diagnostics ----------------------------------------------------------------

struct PatternInsight {
    int questionCount{};
    int exclamationCount{};
    int ellipsisCount{};
    int uppercaseWordCount{};
    int numericTokenCount{};
    int longWordCount{};
    int emailCount{}; // NEW
    int urlCount{};   // NEW
    int hashtagCount{}; // NEW
};

PatternInsight analyzePatterns(const std::string &text, const std::vector<std::string> &words,
                               const std::vector<std::string> &uppercaseWords) {
    PatternInsight insight;

    insight.uppercaseWordCount = static_cast<int>(uppercaseWords.size());

    for (char ch : text) {
        if (ch == '?') {
            ++insight.questionCount;
        } else if (ch == '!') {
            ++insight.exclamationCount;
        }
    }

    std::size_t position = text.find("...");
    while (position != std::string::npos) {
        ++insight.ellipsisCount;
        position = text.find("...", position + 3);
    }

    for (const auto &word : words) {
        if (word.size() >= 10) {
            ++insight.longWordCount;
        }
        if (!word.empty() && std::all_of(word.begin(), word.end(), [](unsigned char ch) {
                return std::isdigit(ch) != 0;
            })) {
            ++insight.numericTokenCount;
        }
    }

    // NEW: Count extracted patterns
    insight.emailCount = extractEmails(text).size();
    insight.urlCount = extractUrls(text).size();
    insight.hashtagCount = extractHashtags(text).size();

    return insight;
}

// --- Text history ----------------------------------------------------------------------------

struct TextHistoryEntry {
    std::string originalInput;
    std::chrono::system_clock::time_point timestamp;
    TextAnalysis analysis; // NEW: Store analysis with history
};

class TextHistory {
  public:
    void addEntry(const std::string &input, const TextAnalysis &analysis) { // CHANGED: Added analysis parameter
        entries.push_back({input, std::chrono::system_clock::now(), analysis});
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
                      << "] \"" << it->originalInput << "\""
                      << " (Words: " << it->analysis.words.size() 
                      << ", Score: " << std::fixed << std::setprecision(2) 
                      << it->analysis.readability.fleschReadingEase << ")" << std::endl;
        }
    }
    
    // NEW: Get analysis from history
    std::optional<TextAnalysis> getAnalysis(std::size_t index) const {
        if (index < entries.size()) {
            auto it = entries.rbegin();
            std::advance(it, index);
            return it->analysis;
        }
        return std::nullopt;
    }
    
    // NEW: Compare current analysis with historical ones
    void printComparison(const TextAnalysis &current) const {
        if (entries.size() < 2) return;
        
        std::cout << "\n📊 Comparison with Previous Analysis" << std::endl;
        std::cout << "-----------------------------------" << std::endl;
        
        const auto &previous = entries.back().analysis;
        
        std::cout << "Word count: " << previous.words.size() << " → " << current.words.size()
                  << " (" << (current.words.size() > previous.words.size() ? "+" : "")
                  << (current.words.size() - previous.words.size()) << ")" << std::endl;
                  
        std::cout << "Readability: " << std::fixed << std::setprecision(1) 
                  << previous.readability.fleschReadingEase << " → " 
                  << current.readability.fleschReadingEase << std::endl;
                  
        std::cout << "Sentiment: " << previous.sentiment.overallFeeling << " → " 
                  << current.sentiment.overallFeeling << std::endl;
    }

  private:
    std::vector<TextHistoryEntry> entries;
    static constexpr std::size_t maximumEntries = 12;
};

// --- Readability profile ---------------------------------------------------------------------

struct ReadabilityProfile {
    double fleschReadingEase{};
    double fleschKincaid{};
    double gunningFog{};
    double smog{};
    double colemanLiau{};
    double automatedReadability{};
    std::string targetAudience; // NEW: Suggested target audience
};

// NEW: Determine target audience based on readability scores
std::string determineTargetAudience(const ReadabilityProfile &profile) {
    if (profile.fleschReadingEase >= 90) return "5th grade students";
    else if (profile.fleschReadingEase >= 80) return "6th grade students";
    else if (profile.fleschReadingEase >= 70) return "7th grade students";
    else if (profile.fleschReadingEase >= 60) return "8th-9th grade students";
    else if (profile.fleschReadingEase >= 50) return "10th-12th grade students";
    else if (profile.fleschReadingEase >= 30) return "College students";
    else return "College graduate level";
}

// --- Text intelligence summary ---------------------------------------------------------------

struct TextAnalysis {
    std::string originalText;
    std::string trimmedText;
    std::string uppercaseText;
    std::string lowercaseText;
    std::string titleCaseText;
    std::string reversedText;

    std::vector<std::string> words;
    std::vector<std::string> uniqueWords;
    std::vector<std::string> sentences;
    std::vector<SentenceInsight> sentenceInsights;

    NGramFrequencyMap bigrams;
    NGramFrequencyMap trigrams;

    CharacterSignature characterSignature;
    SentimentScore sentiment;
    std::vector<KeywordInsight> keywords;
    std::vector<FrequencyEntry> commonPrefixes;
    std::vector<FrequencyEntry> commonSuffixes;
    std::vector<FrequencyEntry> frequentLetterPairs;

    SentenceInsight longestSentence;
    SentenceInsight shortestSentence;

    std::string longestWord;
    std::string shortestWord;

    bool palindrome{};

    int charCountWithSpaces{};
    int charCountWithoutSpaces{};
    int totalSyllables{};
    int totalWordLength{};
    int totalLetters{};
    int complexWordCount{};

    double averageWordLength{};
    double medianWordLength{};
    double vocabularyDiversity{};
    double averageSentenceLength{};
    double averageSyllablesPerWord{};
    double complexWordPercentage{};

    ReadabilityProfile readability;
    PatternInsight patterns;
    AdvancedStats advancedStats; // NEW: Advanced statistics

    std::vector<std::string> uppercaseWords;
    std::vector<std::string> repeatedSentences;
    std::vector<std::string> creativePrompts;
    std::vector<std::string> extractedEmails; // NEW
    std::vector<std::string> extractedUrls;   // NEW
    std::vector<std::string> extractedHashtags; // NEW
};

// --- NEW: Creative writing suggestions -------------------------------------------------------

struct WritingSuggestions {
    std::vector<std::string> vocabularyEnhancements;
    std::vector<std::string> sentenceStructureTips;
    std::vector<std::string> readabilityImprovements;
    std::vector<std::string> engagementBoosters;
};

WritingSuggestions generateWritingSuggestions(const TextAnalysis &analysis) {
    WritingSuggestions suggestions;
    
    // Vocabulary suggestions
    if (analysis.vocabularyDiversity < 0.5) {
        suggestions.vocabularyEnhancements.push_back("Consider using more varied vocabulary to make your writing more engaging");
    }
    
    if (analysis.averageWordLength < 4.0) {
        suggestions.vocabularyEnhancements.push_back("Incorporate more descriptive words to add depth to your writing");
    }
    
    // Sentence structure
    if (analysis.averageSentenceLength > 25) {
        suggestions.sentenceStructureTips.push_back("Break down long sentences for better readability");
    }
    
    if (analysis.averageSentenceLength < 10) {
        suggestions.sentenceStructureTips.push_back("Combine some short sentences to improve flow");
    }
    
    // Readability
    if (analysis.readability.fleschReadingEase < 60) {
        suggestions.readabilityImprovements.push_back("Simplify complex sentences and use more common words");
    }
    
    // Engagement
    if (analysis.patterns.questionCount == 0) {
        suggestions.engagementBoosters.push_back("Consider adding rhetorical questions to engage readers");
    }
    
    if (analysis.sentiment.normalizedScore < -0.2) {
        suggestions.engagementBoosters.push_back("Balance negative content with positive elements");
    }
    
    return suggestions;
}

std::vector<std::string> generateCreativePrompts(const TextAnalysis &analysis) {
    std::vector<std::string> prompts;
    
    if (!analysis.keywords.empty()) {
        std::string prompt = "Write a story featuring: ";
        for (size_t i = 0; i < std::min(size_t(3), analysis.keywords.size()); ++i) {
            if (i > 0) prompt += ", ";
            prompt += analysis.keywords[i].keyword;
        }
        prompts.push_back(prompt);
    }
    
    if (!analysis.longestWord.empty()) {
        prompts.push_back("Create a poem where each line must include the word: " + analysis.longestWord);
    }
    
    if (analysis.sentiment.normalizedScore > 0.3) {
        prompts.push_back("Write a continuation that maintains this positive tone");
    } else if (analysis.sentiment.normalizedScore < -0.3) {
        prompts.push_back("Write a contrasting piece with an optimistic perspective");
    }
    
    prompts.push_back("Rewrite this text from the perspective of a different character");
    prompts.push_back("Create a dialogue based on the key themes in this text");
    
    return prompts;
}

// --- App configuration -----------------------------------------------------------------------

class AppConfig {
  public:
    AppConfig()
        : keywordLimitValue(10), showAsciiChartsValue(true), autoExportEnabledValue(false),
          autoExportPathValue("text_intelligence_report.txt"), highlightUppercaseValue(true),
          enableAdvancedStatsValue(true), saveHistoryValue(true) {} // NEW: Added options

    std::size_t keywordLimit() const { return keywordLimitValue; }
    bool showAsciiCharts() const { return showAsciiChartsValue; }
    bool autoExportEnabled() const { return autoExportEnabledValue; }
    const std::string &autoExportPath() const { return autoExportPathValue; }
    bool highlightUppercase() const { return highlightUppercaseValue; }
    bool enableAdvancedStats() const { return enableAdvancedStatsValue; } // NEW
    bool saveHistory() const { return saveHistoryValue; } // NEW

    void configure() {
        while (true) {
            std::cout << "\n⚙️  Configuration" << std::endl;
            std::cout << "----------------" << std::endl;
            std::cout << "1. Set keyword limit (current: " << keywordLimitValue << ")" << std::endl;
            std::cout << "2. Toggle ASCII charts (current: " << (showAsciiChartsValue ? "On" : "Off") << ")"
                      << std::endl;
            std::cout << "3. Toggle auto-export last report (current: "
                      << (autoExportEnabledValue ? "On" : "Off") << ")" << std::endl;
            std::cout << "4. Change auto-export path (current: " << autoExportPathValue << ")" << std::endl;
            std::cout << "5. Toggle uppercase highlight tips (current: "
                      << (highlightUppercaseValue ? "On" : "Off") << ")" << std::endl;
            std::cout << "6. Toggle advanced statistics (current: "
                      << (enableAdvancedStatsValue ? "On" : "Off") << ")" << std::endl; // NEW
            std::cout << "7. Toggle history saving (current: "
                      << (saveHistoryValue ? "On" : "Off") << ")" << std::endl; // NEW
            std::cout << "8. Return to main menu" << std::endl;
            std::cout << "Select an option (1-8): "; // CHANGED: Updated range

            int choice = 0;
            if (!(std::cin >> choice)) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Invalid input. Please try again." << std::endl;
                continue;
            }

            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

            if (choice == 1) {
                std::cout << "Enter new keyword limit (3 - 25): ";
                std::size_t newLimit = keywordLimitValue;
                if (std::cin >> newLimit && newLimit >= 3 && newLimit <= 25) {
                    keywordLimitValue = newLimit;
                    std::cout << "Keyword limit updated." << std::endl;
                } else {
                    std::cout << "Invalid range. Keeping previous value." << std::endl;
                }
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            } else if (choice == 2) {
                showAsciiChartsValue = !showAsciiChartsValue;
                std::cout << "ASCII charts are now " << (showAsciiChartsValue ? "enabled." : "disabled.")
                          << std::endl;
            } else if (choice == 3) {
                autoExportEnabledValue = !autoExportEnabledValue;
                std::cout << "Auto-export is now " << (autoExportEnabledValue ? "enabled." : "disabled.")
                          << std::endl;
            } else if (choice == 4) {
                std::cout << "Enter new auto-export path: ";
                std::string newPath;
                std::getline(std::cin, newPath);
                if (!newPath.empty()) {
                    autoExportPathValue = newPath;
                    std::cout << "Auto-export path updated." << std::endl;
                } else {
                    std::cout << "Path can't be empty. Keeping previous value." << std::endl;
                }
            } else if (choice == 5) {
                highlightUppercaseValue = !highlightUppercaseValue;
                std::cout << "Uppercase highlight tips are now "
                          << (highlightUppercaseValue ? "enabled." : "disabled.") << std::endl;
            } else if (choice == 6) { // NEW
                enableAdvancedStatsValue = !enableAdvancedStatsValue;
                std::cout << "Advanced statistics are now "
                          << (enableAdvancedStatsValue ? "enabled." : "disabled.") << std::endl;
            } else if (choice == 7) { // NEW
                saveHistoryValue = !saveHistoryValue;
                std::cout << "History saving is now "
                          << (saveHistoryValue ? "enabled." : "disabled.") << std::endl;
            } else if (choice == 8) {
                break;
            } else {
                std::cout << "Unknown option." << std::endl;
            }
        }
    }

  private:
    std::size_t keywordLimitValue;
    bool showAsciiChartsValue;
    bool autoExportEnabledValue;
    std::string autoExportPathValue;
    bool highlightUppercaseValue;
    bool enableAdvancedStatsValue; // NEW
    bool saveHistoryValue; // NEW
};

// --- Core analysis engine --------------------------------------------------------------------

class TextIntelligenceEngine {
  public:
    TextIntelligenceEngine() : sentimentLexicon() {}

    TextAnalysis analyze(const std::string &text, std::size_t keywordLimit) {
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
        analysis.titleCaseText = toTitleCase(text);
        analysis.reversedText = reverseText(text);
        analysis.uppercaseWords = extractUppercaseWords(text);

        analysis.words = splitWords(text);
        analysis.uniqueWords = splitUniqueWords(text);
        analysis.sentences = splitSentences(text);

        analysis.sentenceInsights = analyzeSentences(analysis.sentences);
        analysis.bigrams = buildNGrams(analysis.words, 2);
        analysis.trigrams = buildNGrams(analysis.words, 3);
        analysis.characterSignature = buildCharacterSignature(text);
        analysis.sentiment = analyzeSentiment(analysis.words, sentimentLexicon);
        analysis.keywords = extractTopKeywords(analysis.words, keywordLimit);
        analysis.commonPrefixes = computeAffixFrequency(analysis.words, 3, true, 6);
        analysis.commonSuffixes = computeAffixFrequency(analysis.words, 3, false, 6);
        analysis.frequentLetterPairs = computeLetterPairs(text, 6);

        // NEW: Extract patterns
        analysis.extractedEmails = extractEmails(text);
        analysis.extractedUrls = extractUrls(text);
        analysis.extractedHashtags = extractHashtags(text);

        processWordStatistics(analysis);
        processSentenceStatistics(analysis);
        processPalindrome(analysis);
        processReadability(analysis);
        analysis.patterns = analyzePatterns(text, analysis.words, analysis.uppercaseWords);
        analysis.repeatedSentences = detectRepeatedSentences(analysis.sentences);
        analysis.creativePrompts = generateCreativePrompts(analysis);
        
        // NEW: Advanced statistics
        analysis.advancedStats = computeAdvancedStats(text, analysis.words, analysis.sentences);
        
        // NEW: Find most frequent n-grams
        auto mostFrequentBigram = findMostFrequentNGram(analysis.bigrams);
        auto mostFrequentTrigram = findMostFrequentNGram(analysis.trigrams);
        analysis.advancedStats.mostFrequentBigram = mostFrequentBigram.first;
        analysis.advancedStats.mostFrequentTrigram = mostFrequentTrigram.first;

        return analysis;
    }

    void printAnalysis(const TextAnalysis &analysis, const AppConfig &config) const {
        printHeader();
        printOriginalTextSection(analysis);
        printMeasurementsSection(analysis);
        printReadabilitySection(analysis);
        printSentenceDiagnostics(analysis);
        printKeywordHighlights(analysis);
        printPatternHighlights(analysis, config);
        printNGramHighlights(analysis);
        printCharacterSignature(analysis, config);
        printSentiment(analysis);
        
        // NEW: Print advanced features
        if (config.enableAdvancedStats()) {
            printAdvancedStats(analysis);
            printWritingSuggestions(analysis);
            printExtractedPatterns(analysis);
        }
        
        printRecommendations(analysis, config);
    }

    bool exportAnalysis(const TextAnalysis &analysis, const std::string &path) const {
        std::ofstream file(path);
        if (!file.is_open()) {
            return false;
        }

        file << "Ultra Text Intelligence Report" << std::endl;
        file << "================================" << std::endl;
        file << "Original: " << analysis.originalText << '\n';
        file << "Word count: " << analysis.words.size() << '\n';
        file << "Unique words: " << analysis.uniqueWords.size() << '\n';
        file << "Average word length: " << std::fixed << std::setprecision(2) << analysis.averageWordLength << '\n';
        file << "Flesch Reading Ease: " << std::fixed << std::setprecision(2) << analysis.readability.fleschReadingEase
             << '\n';
        file << "Sentiment: " << analysis.sentiment.overallFeeling
             << " (score: " << std::fixed << std::setprecision(2) << analysis.sentiment.normalizedScore << ")\n";
        file << "Top keywords:" << '\n';
        for (const auto &keyword : analysis.keywords) {
            file << "  - " << keyword.keyword << " (" << keyword.frequency << " occurrences)" << '\n';
        }
        
        // NEW: Export advanced data
        file << "Advanced Statistics:" << '\n';
        file << "  - Paragraphs: " << analysis.advancedStats.paragraphCount << '\n';
        file << "  - Questions: " << analysis.advancedStats.questionCount << '\n';
        file << "  - Exclamations: " << analysis.advancedStats.exclamationCount << '\n';
        file << "  - Character Entropy: " << std::fixed << std::setprecision(3) 
             << analysis.characterSignature.entropy << '\n';
        
        file << std::endl;
        file.close();
        return true;
    }

    SentimentLexicon &lexicon() {
        return sentimentLexicon;
    }

    const SentimentLexicon &lexicon() const {
        return sentimentLexicon;
    }

  private:
    SentimentLexicon sentimentLexicon;

    static void processWordStatistics(TextAnalysis &analysis) {
        analysis.totalSyllables = 0;
        analysis.totalWordLength = 0;
        analysis.totalLetters = 0;
        analysis.complexWordCount = 0;

        std::vector<int> lengths;
        lengths.reserve(analysis.words.size());

        for (const auto &word : analysis.words) {
            const int syllables = estimateSyllables(word);
            analysis.totalSyllables += syllables;
            analysis.totalWordLength += static_cast<int>(word.size());
            analysis.totalLetters += static_cast<int>(word.size());
            lengths.push_back(static_cast<int>(word.size()));

            if (word.size() > analysis.longestWord.size()) {
                analysis.longestWord = word;
            }
            if (analysis.shortestWord.empty() || word.size() < analysis.shortestWord.size()) {
                analysis.shortestWord = word;
            }

            if (syllables >= 3) {
                ++analysis.complexWordCount;
            }
        }

        if (!analysis.words.empty()) {
            analysis.averageWordLength = static_cast<double>(analysis.totalWordLength) /
                                         static_cast<double>(analysis.words.size());
            analysis.vocabularyDiversity = static_cast<double>(analysis.uniqueWords.size()) /
                                           static_cast<double>(analysis.words.size());
            analysis.averageSyllablesPerWord = static_cast<double>(analysis.totalSyllables) /
                                               static_cast<double>(analysis.words.size());
            analysis.complexWordPercentage = static_cast<double>(analysis.complexWordCount) * 100.0 /
                                             static_cast<double>(analysis.words.size());
        }

        if (!lengths.empty()) {
            std::sort(lengths.begin(), lengths.end());
            if (lengths.size() % 2 == 0) {
                const auto mid = lengths.size() / 2;
                analysis.medianWordLength =
                    (static_cast<double>(lengths[mid - 1]) + static_cast<double>(lengths[mid])) / 2.0;
            } else {
                analysis.medianWordLength = static_cast<double>(lengths[lengths.size() / 2]);
            }
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
        analysis.readability.fleschReadingEase = fleschReadingEase(
            static_cast<int>(analysis.words.size()),
            static_cast<int>(analysis.sentences.size()),
            analysis.totalSyllables);

        analysis.readability.fleschKincaid = fleschKincaidGradeLevel(
            static_cast<int>(analysis.words.size()),
            static_cast<int>(analysis.sentences.size()),
            analysis.totalSyllables);

        analysis.readability.gunningFog = gunningFogIndex(
            static_cast<int>(analysis.words.size()),
            static_cast<int>(analysis.sentences.size()),
            analysis.complexWordCount);

        analysis.readability.smog = smogIndex(
            static_cast<int>(analysis.sentences.size()),
            analysis.complexWordCount);

        analysis.readability.colemanLiau = colemanLiauIndex(
            static_cast<int>(analysis.words.size()),
            static_cast<int>(analysis.sentences.size()),
            analysis.totalLetters);

        analysis.readability.automatedReadability = automatedReadabilityIndex(
            static_cast<int>(analysis.words.size()),
            static_cast<int>(analysis.sentences.size()),
            analysis.totalLetters);
            
        // NEW: Determine target audience
        analysis.readability.targetAudience = determineTargetAudience(analysis.readability);
    }

    // Print methods (existing ones would be here, adding new ones for new features)
    
    static void printHeader() {
        std::cout << "\n"
                     "╔══════════════════════════════════════════════════════════════════════════════╗\n"
                     "║                    ULTRA TEXT INTELLIGENCE ANALYSIS                         ║\n"
                     "║                          Enhanced Version 2.0                               ║\n"
                     "╚══════════════════════════════════════════════════════════════════════════════╝\n";
    }

    static void printOriginalTextSection(const TextAnalysis &analysis) {
        std::cout << "\n📝 ORIGINAL TEXT\n";
        std::cout << "================\n";
        std::cout << "Original: \"" << analysis.originalText << "\"\n";
        std::cout << "Trimmed: \"" << analysis.trimmedText << "\"\n";
    }

    static void printMeasurementsSection(const TextAnalysis &analysis) {
        std::cout << "\n📊 BASIC MEASUREMENTS\n";
        std::cout << "=====================\n";
        std::cout << "Characters (with spaces): " << analysis.charCountWithSpaces << "\n";
        std::cout << "Characters (without spaces): " << analysis.charCountWithoutSpaces << "\n";
        std::cout << "Words: " << analysis.words.size() << "\n";
        std::cout << "Unique words: " << analysis.uniqueWords.size() << "\n";
        std::cout << "Sentences: " << analysis.sentences.size() << "\n";
        std::cout << "Vocabulary diversity: " << std::fixed << std::setprecision(2) 
                  << (analysis.vocabularyDiversity * 100.0) << "%\n";
    }

    static void printReadabilitySection(const TextAnalysis &analysis) {
        std::cout << "\n🎓 READABILITY SCORES\n";
        std::cout << "=====================\n";
        std::cout << "Flesch Reading Ease: " << std::fixed << std::setprecision(1) 
                  << analysis.readability.fleschReadingEase << "\n";
        std::cout << "Flesch-Kincaid Grade: " << std::fixed << std::setprecision(1) 
                  << analysis.readability.fleschKincaid << "\n";
        std::cout << "Gunning Fog Index: " << std::fixed << std::setprecision(1) 
                  << analysis.readability.gunningFog << "\n";
        std::cout << "SMOG Index: " << std::fixed << std::setprecision(1) 
                  << analysis.readability.smog << "\n";
        std::cout << "Coleman-Liau Index: " << std::fixed << std::setprecision(1) 
                  << analysis.readability.colemanLiau << "\n";
        std::cout << "Automated Readability: " << std::fixed << std::setprecision(1) 
                  << analysis.readability.automatedReadability << "\n";
        std::cout << "Suggested Audience: " << analysis.readability.targetAudience << "\n";
    }

    // NEW: Print advanced statistics
    static void printAdvancedStats(const TextAnalysis &analysis) {
        std::cout << "\n🔬 ADVANCED STATISTICS\n";
        std::cout << "=====================\n";
        std::cout << "Paragraphs: " << analysis.advancedStats.paragraphCount << "\n";
        std::cout << "Questions: " << analysis.advancedStats.questionCount << "\n";
        std::cout << "Exclamations: " << analysis.advancedStats.exclamationCount << "\n";
        std::cout << "Character Entropy: " << std::fixed << std::setprecision(3) 
                  << analysis.characterSignature.entropy << "\n";
        std::cout << "Word Length Variance: " << std::fixed << std::setprecision(3) 
                  << analysis.advancedStats.wordLengthVariance << "\n";
        std::cout << "Most Frequent Bigram: \"" << analysis.advancedStats.mostFrequentBigram << "\"\n";
        std::cout << "Most Frequent Trigram: \"" << analysis.advancedStats.mostFrequentTrigram << "\"\n";
    }

    // NEW: Print writing suggestions
    static void printWritingSuggestions(const TextAnalysis &analysis) {
        auto suggestions = generateWritingSuggestions(analysis);
        
        std::cout << "\n💡 WRITING SUGGESTIONS\n";
        std::cout << "=====================\n";
        
        if (!suggestions.vocabularyEnhancements.empty()) {
            std::cout << "Vocabulary:\n";
            for (const auto &suggestion : suggestions.vocabularyEnhancements) {
                std::cout << "  • " << suggestion << "\n";
            }
        }
        
        if (!suggestions.sentenceStructureTips.empty()) {
            std::cout << "Sentence Structure:\n";
            for (const auto &suggestion : suggestions.sentenceStructureTips) {
                std::cout << "  • " << suggestion << "\n";
            }
        }
        
        if (!suggestions.readabilityImprovements.empty()) {
            std::cout << "Readability:\n";
            for (const auto &suggestion : suggestions.readabilityImprovements) {
                std::cout << "  • " << suggestion << "\n";
            }
        }
        
        if (!suggestions.engagementBoosters.empty()) {
            std::cout << "Engagement:\n";
            for (const auto &suggestion : suggestions.engagementBoosters) {
                std::cout << "  • " << suggestion << "\n";
            }
        }
    }

    // NEW: Print extracted patterns
    static void printExtractedPatterns(const TextAnalysis &analysis) {
        std::cout << "\n🔍 EXTRACTED PATTERNS\n";
        std::cout << "=====================\n";
        
        if (!analysis.extractedEmails.empty()) {
            std::cout << "Emails found:\n";
            for (const auto &email : analysis.extractedEmails) {
                std::cout << "  • " << email << "\n";
            }
        }
        
        if (!analysis.extractedUrls.empty()) {
            std::cout << "URLs found:\n";
            for (const auto &url : analysis.extractedUrls) {
                std::cout << "  • " << url << "\n";
            }
        }
        
        if (!analysis.extractedHashtags.empty()) {
            std::cout << "Hashtags found:\n";
            for (const auto &hashtag : analysis.extractedHashtags) {
                std::cout << "  • " << hashtag << "\n";
            }
        }
    }

    // Existing print methods would continue here...
    static void printSentenceDiagnostics(const TextAnalysis &analysis) {
        // Implementation...
    }

    static void printKeywordHighlights(const TextAnalysis &analysis) {
        // Implementation...
    }

    static void printPatternHighlights(const TextAnalysis &analysis, const AppConfig &config) {
        // Implementation...
    }

    static void printNGramHighlights(const TextAnalysis &analysis) {
        // Implementation...
    }

    static void printCharacterSignature(const TextAnalysis &analysis, const AppConfig &config) {
        // Implementation...
    }

    static void printSentiment(const TextAnalysis &analysis) {
        // Implementation...
    }

    static void printRecommendations(const TextAnalysis &analysis, const AppConfig &config) {
        // Implementation...
    }
};

// FIXED: Added missing main function and application class
class TextIntelligenceApp {
private:
    TextIntelligenceEngine engine;
    AppConfig config;
    TextHistory history;
    std::vector<std::vector<std::string>> documentCorpus; // For TF-IDF

public:
    void run() {
        while (true) {
            displayMainMenu();
            int choice = getMenuChoice(1, 7);
            
            switch (choice) {
                case 1: analyzeText(); break;
                case 2: viewHistory(); break;
                case 3: compareAnalyses(); break;
                case 4: manageLexicon(); break;
                case 5: config.configure(); break;
                case 6: showHelp(); break;
                case 7: return;
            }
        }
    }

private:
    void displayMainMenu() {
        std::cout << "\n"
                     "╔══════════════════════════════════════════════════════════════════════════════╗\n"
                     "║                         TEXT INTELLIGENCE ANALYZER                          ║\n"
                     "║                               Version 2.0                                   ║\n"
                     "╚══════════════════════════════════════════════════════════════════════════════╝\n";
        std::cout << "1. Analyze Text\n";
        std::cout << "2. View History\n";
        std::cout << "3. Compare Analyses\n";
        std::cout << "4. Manage Sentiment Lexicon\n";
        std::cout << "5. Configuration\n";
        std::cout << "6. Help & Information\n";
        std::cout << "7. Exit\n";
        std::cout << "Select an option (1-7): ";
    }

    int getMenuChoice(int min, int max) {
        int choice;
        while (!(std::cin >> choice) || choice < min || choice > max) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid choice. Please enter a number between " << min << " and " << max << ": ";
        }
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        return choice;
    }

    void analyzeText() {
        std::cout << "\nEnter text to analyze (or 'back' to return):\n> ";
        std::string input;
        std::getline(std::cin, input);
        
        if (input == "back") return;
        if (input.empty()) {
            std::cout << "No text provided.\n";
            return;
        }

        auto analysis = engine.analyze(input, config.keywordLimit());
        
        // Add to corpus for TF-IDF
        documentCorpus.push_back(analysis.words);
        
        // Update analysis with TF-IDF keywords
        analysis.keywords = extractTopKeywordsWithTFIDF(analysis.words, documentCorpus, config.keywordLimit());
        
        engine.printAnalysis(analysis, config);
        
        if (config.saveHistory()) {
            history.addEntry(input, analysis);
        }
        
        if (config.autoExportEnabled()) {
            if (engine.exportAnalysis(analysis, config.autoExportPath())) {
                std::cout << "Report exported to: " << config.autoExportPath() << "\n";
            } else {
                std::cout << "Failed to export report.\n";
            }
        }
    }

    void viewHistory() {
        history.printHistory();
    }

    void compareAnalyses() {
        if (!config.saveHistory()) {
            std::cout << "History saving is disabled. Enable it in configuration.\n";
            return;
        }
        
        std::cout << "\nEnter the index of the analysis to compare (0 for most recent): ";
        int index;
        std::cin >> index;
        std::cin.ignore();
        
        auto historical = history.getAnalysis(index);
        if (!historical) {
            std::cout << "Invalid index.\n";
            return;
        }
        
        std::cout << "\nEnter new text to compare:\n> ";
        std::string newText;
        std::getline(std::cin, newText);
        
        auto newAnalysis = engine.analyze(newText, config.keywordLimit());
        history.printComparison(newAnalysis);
    }

    void manageLexicon() {
        while (true) {
            std::cout << "\n📚 Sentiment Lexicon Management\n";
            std::cout << "1. View Lexicon Summary\n";
            std::cout << "2. Add Positive Word\n";
            std::cout << "3. Add Negative Word\n";
            std::cout << "4. Remove Word\n";
            std::cout << "5. Load from File\n";
            std::cout << "6. Save to File\n";
            std::cout << "7. Back to Main Menu\n";
            std::cout << "Select option (1-7): ";
            
            int choice = getMenuChoice(1, 7);
            
            switch (choice) {
                case 1: engine.lexicon().printSummary(); break;
                case 2: addWordToLexicon(true); break;
                case 3: addWordToLexicon(false); break;
                case 4: removeWordFromLexicon(); break;
                case 5: loadLexiconFromFile(); break;
                case 6: saveLexiconToFile(); break;
                case 7: return;
            }
        }
    }

    void addWordToLexicon(bool positive) {
        std::cout << "Enter word to add: ";
        std::string word;
        std::getline(std::cin, word);
        
        if (positive) {
            engine.lexicon().addPositive(word);
            std::cout << "Added '" << word << "' to positive words.\n";
        } else {
            engine.lexicon().addNegative(word);
            std::cout << "Added '" << word << "' to negative words.\n";
        }
    }

    void removeWordFromLexicon() {
        std::cout << "Enter word to remove: ";
        std::string word;
        std::getline(std::cin, word);
        
        bool removed = engine.lexicon().removePositive(word) || engine.lexicon().removeNegative(word);
        if (removed) {
            std::cout << "Removed '" << word << "' from lexicon.\n";
        } else {
            std::cout << "Word '" << word << "' not found in lexicon.\n";
        }
    }

    void loadLexiconFromFile() {
        std::cout << "Enter filename: ";
        std::string filename;
        std::getline(std::cin, filename);
        
        if (engine.lexicon().loadFromFile(filename)) {
            std::cout << "Lexicon loaded successfully.\n";
        } else {
            std::cout << "Failed to load lexicon from file.\n";
        }
    }

    void saveLexiconToFile() {
        std::cout << "Enter filename: ";
        std::string filename;
        std::getline(std::cin, filename);
        
        if (engine.lexicon().saveToFile(filename)) {
            std::cout << "Lexicon saved successfully.\n";
        } else {
            std::cout << "Failed to save lexicon to file.\n";
        }
    }

    void showHelp() {
        std::cout << "\n📖 HELP & INFORMATION\n";
        std::cout << "=====================\n";
        std::cout << "This tool analyzes text for:\n";
        std::cout << "• Readability scores (Flesch, Gunning Fog, etc.)\n";
        std::cout << "• Sentiment analysis\n";
        std::cout << "• Keyword extraction and frequency analysis\n";
        std::cout << "• Pattern detection (emails, URLs, hashtags)\n";
        std::cout << "• Advanced statistics and writing suggestions\n";
        std::cout << "\nUse the configuration menu to customize analysis settings.\n";
    }
};

} // namespace text_utils

int main() {
    text_utils::TextIntelligenceApp app;
    app.run();
    return 0;
}