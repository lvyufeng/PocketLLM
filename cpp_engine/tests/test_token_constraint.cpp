#include "token_constraint.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace pocket;

namespace {

std::unique_ptr<bool[]> make_mask(size_t size) {
    auto mask = std::make_unique<bool[]>(size);
    for (size_t i = 0; i < size; ++i) mask[i] = false;
    return mask;
}

std::vector<std::string> vocabulary() {
    return {
        "<eos>", "{", "\"", "key", "\":", "\"value\"", "}", ",",
        "123", "{\"", "\":\"", "\"}", "name", "age",
    };
}

std::vector<uint8_t> special_tokens() {
    return {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
}

}  // namespace

void test_basic_masking() {
    auto constraint = make_json_object_constraint(vocabulary(), special_tokens());
    auto mask = make_mask(14);
    constraint->fill_mask(mask.get(), 0, 14);

    assert(mask[1]);
    assert(mask[9]);
    assert(!mask[0]);
    assert(!mask[2]);
    assert(!mask[6]);
}

void test_accept_token_advances() {
    auto constraint = make_json_object_constraint(vocabulary(), special_tokens());
    assert(constraint->accept_token(1));

    auto mask = make_mask(14);
    constraint->fill_mask(mask.get(), 0, 14);
    assert(mask[2]);
    assert(!mask[0]);

    assert(constraint->accept_token(2));
    assert(!constraint->is_complete());
}

void test_multi_character_pieces() {
    auto constraint = make_json_object_constraint(vocabulary(), special_tokens());

    // The pieces cross several JSON boundaries: { + key quote, key quote + colon
    // + value quote, and value quote + object close.
    assert(constraint->accept_token(9));
    assert(constraint->accept_token(12));
    assert(constraint->accept_token(10));
    assert(constraint->accept_token(12));
    assert(constraint->accept_token(11));
    assert(constraint->is_complete());
}

void test_completion_and_special_tokens() {
    auto constraint = make_json_object_constraint(vocabulary(), special_tokens());
    auto mask = make_mask(14);
    constraint->fill_mask(mask.get(), 0, 14);
    assert(!mask[0]);

    assert(constraint->accept_token(1));
    assert(constraint->accept_token(2));
    assert(constraint->accept_token(3));
    assert(constraint->accept_token(4));
    assert(constraint->accept_token(5));
    assert(constraint->accept_token(6));
    assert(constraint->is_complete());

    constraint->fill_mask(mask.get(), 0, 14);
    assert(!mask[0]);
    assert(!constraint->accept_token(0));
    assert(!constraint->accept_token(7));
}

void test_greedy_masked_walk() {
    auto constraint = make_json_object_constraint(vocabulary(), special_tokens());
    const std::vector<int> preferred = {1, 2, 3, 4, 5, 6};
    std::vector<int> generated;
    generated.reserve(preferred.size());

    for (int token_id : preferred) {
        auto mask = make_mask(14);
        constraint->fill_mask(mask.get(), 0, 14);
        assert(mask[token_id]);
        assert(constraint->accept_token(token_id));
        generated.push_back(token_id);
    }

    assert(constraint->is_complete());
    std::string json;
    const auto pieces = vocabulary();
    for (int token_id : generated) json += pieces[static_cast<size_t>(token_id)];
    assert(json == "{\"key\":\"value\"}");
}

void test_invalid_token_does_not_advance() {
    auto constraint = make_json_object_constraint(vocabulary(), special_tokens());
    assert(constraint->accept_token(1));
    assert(!constraint->accept_token(7));

    auto mask = make_mask(14);
    constraint->fill_mask(mask.get(), 0, 14);
    assert(mask[2]);
    assert(constraint->accept_token(2));
}

int main() {
    test_basic_masking();
    test_accept_token_advances();
    test_multi_character_pieces();
    test_completion_and_special_tokens();
    test_greedy_masked_walk();
    test_invalid_token_does_not_advance();
    std::cout << "All token constraint tests passed\n";
    return 0;
}
