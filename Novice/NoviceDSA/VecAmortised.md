# Vector Amortised Time
- Amortised time refers to a case where very occasionally an operation may take longer, but usually doesn't
    - For example, vector append is usually O(1), but if the vector runs out of space, we create a new vector of double size and copy everything over from the old to the new vector. This takes O(n), but happens rarely.
    - Hence, we consider the amortised time complexity as O(1) because this is usually the case.