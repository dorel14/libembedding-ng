#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:?Usage: $0 <version>}"
MAJOR="${VERSION%%.*}"
MINOR_PATCH="${VERSION#*.}"
MINOR="${MINOR_PATCH%%.*}"
PATCH="${MINOR_PATCH#*.}"

echo "Syncing version ${VERSION} across C/Python headers..."

# Update config.h (C API version macros + header comment)
sed -i 's/^ \* Version: [0-9][0-9.]*/ * Version: '"${VERSION}"'/' include/libembedding/config.h
sed -i 's/^#define LIBEMBEDDING_VERSION_MAJOR [0-9]*$/#define LIBEMBEDDING_VERSION_MAJOR '"${MAJOR}"'/' include/libembedding/config.h
sed -i 's/^#define LIBEMBEDDING_VERSION_MINOR [0-9]*$/#define LIBEMBEDDING_VERSION_MINOR '"${MINOR}"'/' include/libembedding/config.h
sed -i 's/^#define LIBEMBEDDING_VERSION_PATCH [0-9]*$/#define LIBEMBEDDING_VERSION_PATCH '"${PATCH}"'/' include/libembedding/config.h
sed -i 's/^#define LIBEMBEDDING_VERSION_STRING "[0-9.]*"$/#define LIBEMBEDDING_VERSION_STRING "'"${VERSION}"'"/' include/libembedding/config.h

# Update CMakeLists.txt project VERSION
sed -i 's/^    VERSION [0-9][0-9.]*$/    VERSION '"${VERSION}"'/' CMakeLists.txt

# Update all C/C++ header version comments
find include -type f \( -name "*.h" -o -name "*.hpp" \) -not -path "*/third_party/*" | while read -r f; do
    sed -i 's/^ \* Version: [0-9][0-9.]*/ * Version: '"${VERSION}"'/' "$f"
done

# Update Python module version comments
find python/src -type f -name "*.py" | while read -r f; do
    sed -i 's/^Version: [0-9][0-9.]*/Version: '"${VERSION}"'/' "$f"
done

# Update _cdefs.h synced version note
sed -i 's/(v[0-9][0-9.]*)/(v'"${VERSION}"')/' python/src/libembedding/_cdefs.h

# Update AGENTS.md current version line
sed -i 's/^> \*\*Version courante\*\* : [0-9][0-9.]*$/> **Version courante** : '"${VERSION}"'/' AGENTS.md

echo "Version sync complete."
