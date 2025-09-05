#pragma once

#include <nlohmann/json.hpp>
#include <any>
#include <map>
#include <vector>
#include <string>
#include <stdexcept>
#include <cctype>
#include <fmt/core.h>
#include <fmt/color.h>

using json = nlohmann::json;
using AnyMap = std::map<std::string, std::any>;

// ---- Forward decls ----
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

inline AnyMap convertJsonObject(const json &obj)
{
    AnyMap m;
    for (auto it = obj.begin(); it != obj.end(); ++it)
    {
        if (it.value().is_null())
        {
            m[it.key()] = nullptr;
            continue;
        }
        m[it.key()] = convertJsonToAny(it.value());
    }
    return m;
}

inline int convertJsonToMap(const json &j, AnyMap &result)
{
    try
    {
        if (!j.is_object())
            throw std::runtime_error("Top-level JSON must be an object");
        result = convertJsonObject(j);
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
    else if (j.is_object())
        return convertJsonObject(j); // <-- nested objects
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
        throw std::runtime_error(std::string("Missing key: ") + key);
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

// Fetch exact type (e.g. string, bool, nested AnyMap) — no conversion
template <typename T>
const T &get_exact(const AnyMap &m, const char *key)
{
    return std::any_cast<const T &>(must_get(m, key));
}

// Non-const overload if needed
template <typename T>
T &get_exact(AnyMap &m, const char *key)
{
    auto &a = const_cast<std::any &>(must_get(m, key));
    return std::any_cast<T &>(a);
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

// Fetch array of nested maps (vector<any> → vector<AnyMap>)
inline std::vector<AnyMap> get_map_array(const AnyMap &m, const char *key)
{
    const auto &a = must_get(m, key);
    const auto *pv = std::any_cast<std::vector<std::any>>(&a);
    if (!pv)
        throw std::bad_any_cast();
    std::vector<AnyMap> out;
    out.reserve(pv->size());
    for (const auto &e : *pv)
    {
        const auto *pm = std::any_cast<AnyMap>(&e);
        if (!pm)
            throw std::bad_any_cast();
        out.push_back(*pm);
    }
    return out;
}

// ---------------- Nested Path Access (dot + [index]) ---------------- //
// Simple path format, examples:
//   "camera.lens.focal"
//   "centers[0]" or "camera.centers[1]"
// Notes: indexes apply to vector<std::any>; no quotes/escapes in keys.

namespace detail
{

    // Return (segment, index_opt) where segment may have [idx] suffix (single, optional)
    struct PathSeg
    {
        std::string key;
        bool has_index{false};
        size_t index{0};
    };

    inline PathSeg parse_segment(const std::string &seg)
    {
        PathSeg r;
        r.key = seg;
        auto lb = seg.find('[');
        if (lb != std::string::npos)
        {
            auto rb = seg.find(']', lb + 1);
            if (rb == std::string::npos)
                throw std::runtime_error("Malformed path segment: " + seg);
            r.key = seg.substr(0, lb);
            std::string idxs = seg.substr(lb + 1, rb - lb - 1);
            if (idxs.empty())
                throw std::runtime_error("Empty index in path segment: " + seg);
            for (char c : idxs)
                if (!std::isdigit(static_cast<unsigned char>(c)))
                    throw std::runtime_error("Non-numeric index in path segment: " + seg);
            r.has_index = true;
            r.index = static_cast<size_t>(std::stoull(idxs));
        }
        return r;
    }

    inline std::vector<std::string> split_path(const std::string &path)
    {
        std::vector<std::string> parts;
        std::string cur;
        for (char c : path)
        {
            if (c == '.')
            {
                if (!cur.empty())
                {
                    parts.push_back(cur);
                    cur.clear();
                }
            }
            else
                cur.push_back(c);
        }
        if (!cur.empty())
            parts.push_back(cur);
        if (parts.empty())
            throw std::runtime_error("Empty path");
        return parts;
    }

} // namespace detail

inline const std::any &must_get_path(const AnyMap &root, const std::string &path)
{
    const AnyMap *cur_map = &root;
    const std::any *cur_any = nullptr;

    auto parts = detail::split_path(path);
    for (size_t i = 0; i < parts.size(); ++i)
    {
        auto seg = detail::parse_segment(parts[i]);

        // map lookup
        auto it = cur_map->find(seg.key);
        if (it == cur_map->end())
            throw std::runtime_error("Missing key in path: " + seg.key);
        cur_any = &it->second;

        // optional index: treat value as vector<any>
        if (seg.has_index)
        {
            const auto *pv = std::any_cast<std::vector<std::any>>(cur_any);
            if (!pv)
                throw std::bad_any_cast();
            if (seg.index >= pv->size())
                throw std::out_of_range("Index out of range in path segment: " + parts[i]);
            cur_any = &(*pv)[seg.index];
        }

        // If not at last segment and current value is a map, descend
        if (i + 1 < parts.size())
        {
            if (const auto *pm = std::any_cast<AnyMap>(cur_any))
            {
                cur_map = pm;
            }
            else
            {
                // Not a map → next step invalid
                throw std::bad_any_cast();
            }
        }
    }
    return *cur_any;
}

// Nested getters using path syntax
template <typename T>
T get_num_path(const AnyMap &root, const std::string &path)
{
    return numeric_cast_from_any<T>(must_get_path(root, path));
}

template <typename T>
const T &get_exact_path(const AnyMap &root, const std::string &path)
{
    return std::any_cast<const T &>(must_get_path(root, path));
}

inline const std::string &get_str_path(const AnyMap &root, const std::string &path)
{
    return get_exact_path<std::string>(root, path);
}

template <typename T>
std::vector<T> get_num_array_path(const AnyMap &root, const std::string &path)
{
    const auto &a = must_get_path(root, path);
    const auto *pv = std::any_cast<std::vector<std::any>>(&a);
    if (!pv)
        throw std::bad_any_cast();
    std::vector<T> out;
    out.reserve(pv->size());
    for (const auto &e : *pv)
        out.push_back(numeric_cast_from_any<T>(e));
    return out;
}

inline const AnyMap &get_map_path(const AnyMap &root, const std::string &path)
{
    return std::any_cast<const AnyMap &>(must_get_path(root, path));
}

inline std::vector<AnyMap> get_map_array_path(const AnyMap &root, const std::string &path)
{
    const auto &a = must_get_path(root, path);
    const auto *pv = std::any_cast<std::vector<std::any>>(&a);
    if (!pv)
        throw std::bad_any_cast();
    std::vector<AnyMap> out;
    out.reserve(pv->size());
    for (const auto &e : *pv)
    {
        const auto *pm = std::any_cast<AnyMap>(&e);
        if (!pm)
            throw std::bad_any_cast();
        out.push_back(*pm);
    }
    return out;
}
