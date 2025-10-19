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
            "nice", "optimistic", "positive", "smart", "terrific", "vibrant", "wonderful"};

        negativeWords = {
            "awful", "bad", "boring", "broken", "cruel", "damaged", "dark", "depressing", "evil", "fail",
            "gloomy", "horrible", "hurt", "imperfect", "jealous", "lonely", "mad", "negative", "pain",
            "sad", "terrible", "ugly", "worst", "worthless"};
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

  private:
    std::unordered_set<std::string> positiveWords;
    std::unordered_set<std::string> negativeWords;
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

// --- Patterns and diagnostics ----------------------------------------------------------------

struct PatternInsight {
    int questionCount{};
    int exclamationCount{};
    int ellipsisCount{};
    int uppercaseWordCount{};
    int numericTokenCount{};
    int longWordCount{};
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

    return insight;
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

// --- Readability profile ---------------------------------------------------------------------

struct ReadabilityProfile {
    double fleschReadingEase{};
    double fleschKincaid{};
    double gunningFog{};
    double smog{};
    double colemanLiau{};
    double automatedReadability{};
};

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

    std::vector<std::string> uppercaseWords;
    std::vector<std::string> repeatedSentences;
    std::vector<std::string> creativePrompts;
};

// --- App configuration -----------------------------------------------------------------------

class AppConfig {
  public:
    AppConfig()
        : keywordLimitValue(10), showAsciiChartsValue(true), autoExportEnabledValue(false),
          autoExportPathValue("text_intelligence_report.txt"), highlightUppercaseValue(true) {}

    std::size_t keywordLimit() const {
        return keywordLimitValue;
    }

    bool showAsciiCharts() const {
        return showAsciiChartsValue;
    }

    bool autoExportEnabled() const {
        return autoExportEnabledValue;
    }

    const std::string &autoExportPath() const {
        return autoExportPathValue;
    }

    bool highlightUppercase() const {
        return highlightUppercaseValue;
    }

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
            std::cout << "6. Return to main menu" << std::endl;
            std::cout << "Select an option (1-6): ";

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
            } else if (choice == 6) {
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

        processWordStatistics(analysis);
        processSentenceStatistics(analysis);
        processPalindrome(analysis);
        processReadability(analysis);
        analysis.patterns = analyzePatterns(text, analysis.words, analysis.uppercaseWords);
        analysis.repeatedSentences = detectRepeatedSentences(analysis.sentences);
        analysis.creativePrompts = generateCreativePrompts(analysis);

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

    static void