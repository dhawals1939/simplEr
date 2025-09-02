#pragma once

#include <nlohmann/json.hpp>
#include <any>
#include <map>
#include <vector>
#include <string>
#include <stdexcept>
#include <fmt/core.h>
#include <fmt/color.h>

using json = nlohmann::json;
using AnyMap = std::map<std::string, std::any>;

// Forward declaration
std::any convertJsonToAny(const json &j);

// ---------------- Conversion Helpers ---------------- //

inline std::vector<std::any> convertJsonArray(const json &arr)
{
    std::vector<std::any> result;
    result.reserve(arr.size());
    for (const auto &item : arr)
    {
        result.push_back(convertJsonToAny(item));
    }
    return result;
}

inline int convertJsonToMap(const json &j, AnyMap &result)
{
    try
    {
        for (auto it = j.begin(); it != j.end(); ++it)
        {
            if (it.value().is_null())
            {
                result[it.key()] = nullptr;
                continue;
            }
            result[it.key()] = convertJsonToAny(it.value());
        }
    }
    catch (const std::exception &e)
    {
        fmt::print(fmt::fg(fmt::color::red) | fmt::emphasis::bold,
                   "Error: {}\n", e.what());
        result.clear();
        result["status"] = "error";
        return -1;
    }
    return 0;
}

inline std::any convertJsonToAny(const json &j)
{
    if (j.is_array())
        return convertJsonArray(j);
    else if (j.is_string())
        return j.get<std::string>();
    else if (j.is_number_float())
        return j.get<double>();
    else if (j.is_number_integer())
        return j.get<long long>();
    else if (j.is_number_unsigned())
        return j.get<unsigned long long>();
    else if (j.is_boolean())
        return j.get<bool>();
    else if (j.is_null())
        return nullptr;
    throw std::runtime_error("Unsupported JSON type");
}

// ---------------- Accessor Utilities ---------------- //

// Safer lookup: throws if key is missing
inline const std::any &must_get(const AnyMap &m, const char *key)
{
    auto it = m.find(key);
    if (it == m.end())
    {
        throw std::runtime_error(std::string("Missing key: ") + key);
    }
    return it->second;
}

// Convert any numeric type in std::any to requested T
template <typename T>
T numeric_cast_from_any(const std::any &a)
{
    if (auto p = std::any_cast<T>(&a))
        return *p;
    if (auto p = std::any_cast<long long>(&a))
        return static_cast<T>(*p);
    if (auto p = std::any_cast<unsigned long long>(&a))
        return static_cast<T>(*p);
    if (auto p = std::any_cast<double>(&a))
        return static_cast<T>(*p);
    if (auto p = std::any_cast<int>(&a))
        return static_cast<T>(*p);
    if (auto p = std::any_cast<bool>(&a))
        return static_cast<T>(*p);
    throw std::bad_any_cast();
}

// Fetch numeric value as T (int, float, double, etc.)
template <typename T>
T get_num(const AnyMap &m, const char *key)
{
    return numeric_cast_from_any<T>(must_get(m, key));
}

// Fetch exact type (e.g. string, bool) — no conversion
template <typename T>
T get_exact(const AnyMap &m, const char *key)
{
    return std::any_cast<T>(must_get(m, key));
}

// Fetch numeric array (vector<any> → vector<T>)
template <typename T>
std::vector<T> get_num_array(const AnyMap &m, const char *key)
{
    const auto &a = must_get(m, key);
    const auto *pv = std::any_cast<std::vector<std::any>>(&a);
    if (!pv)
        throw std::bad_any_cast();
    std::vector<T> out;
    out.reserve(pv->size());
    for (const auto &e : *pv)
        out.push_back(numeric_cast_from_any<T>(e));
    return out;
}
