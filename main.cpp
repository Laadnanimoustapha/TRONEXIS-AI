#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace text_utils {

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

std::string normalizeForPalindrome(const std::string &text) {
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

struct TextAnalysis {
    std::string originalText;
    std::string trimmedText;
    std::string uppercaseText;
    std::string lowercaseText;
    std::size_t charCountWithSpaces{};
    std::size_t charCountWithoutSpaces{};
    std::size_t wordCount{};
    std::string longestWord;
    bool palindrome{};
    std::map<char, int> letterFrequency;
};

TextAnalysis analyze(const std::string &text) {
    TextAnalysis result;
    result.originalText = text;
    result.trimmedText = trim(text);
    result.uppercaseText = toUpper(text);
    result.lowercaseText = toLower(text);
    result.charCountWithSpaces = text.size();

    result.charCountWithoutSpaces = static_cast<std::size_t>(std::count_if(
        text.begin(), text.end(), [](unsigned char ch) {
            return std::isspace(ch) == 0;
        }));

    std::istringstream stream(result.trimmedText);
    std::string word;
    while (stream >> word) {
        ++result.wordCount;
        if (word.size() > result.longestWord.size()) {
            result.longestWord = word;
        }

        for (unsigned char ch : word) {
            if (std::isalpha(ch) != 0) {
                result.letterFrequency[static_cast<char>(std::toupper(ch))]++;
            }
        }
    }

    const std::string normalized = normalizeForPalindrome(result.trimmedText);
    std::string reversedNormalized = normalized;
    std::reverse(reversedNormalized.begin(), reversedNormalized.end());
    result.palindrome = normalized == reversedNormalized && !normalized.empty();

    return result;
}

void printAnalysis(const TextAnalysis &analysis) {
    std::cout << "\n🧠 Text Intelligence Report" << std::endl;
    std::cout << "────────────────────────────" << std::endl;
    std::cout << "Original: " << analysis.originalText << std::endl;
    std::cout << "Trimmed: " << analysis.trimmedText << std::endl;
    std::cout << "Uppercase: " << analysis.uppercaseText << std::endl;
    std::cout << "Lowercase: " << analysis.lowercaseText << std::endl;

    std::cout << "\n📏 Measurements" << std::endl;
    std::cout << "- Characters (with spaces): " << analysis.charCountWithSpaces << std::endl;
    std::cout << "- Characters (without spaces): " << analysis.charCountWithoutSpaces << std::endl;
    std::cout << "- Word count: " << analysis.wordCount << std::endl;
    std::cout << "- Longest word: "
              << (analysis.longestWord.empty() ? "<none>" : analysis.longestWord) << std::endl;
    std::cout << "- Palindrome (ignoring punctuation & case): "
              << (analysis.palindrome ? "Yes" : "No") << std::endl;

    std::cout << "\n🔡 Letter frequency" << std::endl;
    if (analysis.letterFrequency.empty()) {
        std::cout << "No alphabetical characters detected." << std::endl;
    } else {
        for (const auto &[letter, count] : analysis.letterFrequency) {
            std::cout << "  " << letter << ": " << count << std::endl;
        }
    }
}

} // namespace text_utils

int main() {
    std::cout << "✨ Welcome to the Text Intelligence Console ✨" << std::endl;
    std::cout << "Type any sentence and I will reveal its secrets." << std::endl;
    std::cout << "Enter 'exit' to leave the program." << std::endl;

    std::string userText;
    while (true) {
        std::cout << "\nPlease enter your text: ";
        if (!std::getline(std::cin, userText)) {
            std::cout << "\nInput stream closed. Goodbye!" << std::endl;
            return 0;
        }

        if (text_utils::equalsIgnoreCase(userText, "exit")) {
            std::cout << "Thanks for exploring text intelligence. 👋" << std::endl;
            break;
        }

        const auto analysis = text_utils::analyze(userText);
        text_utils::printAnalysis(analysis);
    }

    return 0;
}