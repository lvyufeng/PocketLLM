#include "token_constraint.hpp"

#include "json_constraint.hpp"
#include "json_lite.hpp"
#include "tokenizer.hpp"

#include <unordered_map>
#include <utility>

namespace pocket {

namespace {

class JsonTokenConstraint final : public TokenConstraint {
public:
    JsonTokenConstraint(JsonConstraint constraint,
                        std::vector<std::string> token_pieces,
                        std::vector<uint8_t> special_token_ids)
        : constraint_(std::move(constraint)),
          token_pieces_(std::move(token_pieces)),
          special_token_ids_(std::move(special_token_ids)) {}

    void fill_mask(bool* allowed, int vocab_begin, int vocab_end) const override {
        if (allowed == nullptr || vocab_end < vocab_begin) return;
        for (int token_id = vocab_begin; token_id < vocab_end; ++token_id) {
            const int index = token_id - vocab_begin;
            if (token_id < 0 ||
                static_cast<size_t>(token_id) >= token_pieces_.size() ||
                (static_cast<size_t>(token_id) < special_token_ids_.size() &&
                 special_token_ids_[static_cast<size_t>(token_id)] != 0)) {
                allowed[index] = false;
                continue;
            }

            const auto cached = cache_.find(token_id);
            if (cached != cache_.end()) {
                allowed[index] = cached->second;
                continue;
            }
            const std::string& piece = token_pieces_[static_cast<size_t>(token_id)];
            const bool valid = !piece.empty() && constraint_.can_accept(piece);
            cache_.emplace(token_id, valid);
            allowed[index] = valid;
        }
    }

    bool accept_token(int token_id) override {
        if (token_id < 0 || static_cast<size_t>(token_id) >= token_pieces_.size()) {
            return false;
        }
        if (static_cast<size_t>(token_id) < special_token_ids_.size() &&
            special_token_ids_[static_cast<size_t>(token_id)] != 0) {
            return false;
        }
        const std::string& piece = token_pieces_[static_cast<size_t>(token_id)];
        if (piece.empty() || !constraint_.accept(piece)) return false;
        cache_.clear();
        return true;
    }

    bool is_complete() const override { return constraint_.is_complete(); }

    void reset() override {
        constraint_.reset();
        cache_.clear();
    }

private:
    mutable JsonConstraint constraint_;
    std::vector<std::string> token_pieces_;
    std::vector<uint8_t> special_token_ids_;
    mutable std::unordered_map<int, bool> cache_;
};

std::vector<std::string> vocabulary_pieces(const Tokenizer& tokenizer) {
    std::vector<std::string> pieces;
    pieces.reserve(tokenizer.vocab_size());
    for (size_t id = 0; id < tokenizer.vocab_size(); ++id) {
        pieces.push_back(tokenizer.decode_piece(static_cast<int>(id)));
    }
    return pieces;
}

std::vector<uint8_t> special_token_ids(const Tokenizer& tokenizer) {
    std::vector<uint8_t> special(tokenizer.vocab_size(), 0);
    for (size_t id = 0; id < tokenizer.vocab_size(); ++id) {
        special[id] = tokenizer.is_special_token(static_cast<int>(id)) ? 1 : 0;
    }
    return special;
}

}  // namespace

std::unique_ptr<TokenConstraint> make_json_object_constraint(
    const Tokenizer& tokenizer) {
    return std::make_unique<JsonTokenConstraint>(
        JsonConstraint::json_object(), vocabulary_pieces(tokenizer),
        special_token_ids(tokenizer));
}

std::unique_ptr<TokenConstraint> make_json_schema_constraint(
    const Tokenizer& tokenizer, const JsonValue& schema) {
    return std::make_unique<JsonTokenConstraint>(
        JsonConstraint::from_schema(schema), vocabulary_pieces(tokenizer),
        special_token_ids(tokenizer));
}

std::unique_ptr<TokenConstraint> make_json_object_constraint(
    std::vector<std::string> token_pieces,
    std::vector<uint8_t> special_token_ids) {
    return std::make_unique<JsonTokenConstraint>(
        JsonConstraint::json_object(), std::move(token_pieces),
        std::move(special_token_ids));
}

}  // namespace pocket
