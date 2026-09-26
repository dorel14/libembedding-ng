# FindONNXRuntime.cmake
# Locates ONNX Runtime installation
#
# Sets:
#   ONNXRuntime_FOUND
#   ONNXRuntime_INCLUDE_DIRS
#   ONNXRuntime_LIBRARIES
#
# Creates imported target: ONNXRuntime::ONNXRuntime
#
# Hints:
#   ONNXRUNTIME_ROOT (env or cmake var)

# Search paths
set(_onnxruntime_search_paths "")

if(DEFINED ENV{ONNXRUNTIME_ROOT})
    list(APPEND _onnxruntime_search_paths "$ENV{ONNXRUNTIME_ROOT}")
endif()

if(DEFINED ONNXRUNTIME_ROOT)
    list(APPEND _onnxruntime_search_paths "${ONNXRUNTIME_ROOT}")
endif()

# Common install locations
list(APPEND _onnxruntime_search_paths
    "/usr/local"
    "/usr"
    "/opt/onnxruntime"
    "$ENV{HOME}/.local"
)

# macOS Homebrew
if(APPLE)
    list(APPEND _onnxruntime_search_paths
        "/opt/homebrew"
        "/opt/homebrew/opt/onnxruntime"
    )
endif()

# Find include directory
find_path(ONNXRuntime_INCLUDE_DIR
    NAMES onnxruntime_c_api.h
    PATHS ${_onnxruntime_search_paths}
    PATH_SUFFIXES include include/onnxruntime
)

# Find library - search for both unversioned and versioned names
find_library(ONNXRuntime_LIBRARY
    NAMES onnxruntime
    PATHS ${_onnxruntime_search_paths}
    PATH_SUFFIXES lib lib64
)

# Fallback: search for versioned shared libraries (e.g. libonnxruntime.so.1.20.1)
if(NOT ONNXRuntime_LIBRARY)
    foreach(_dir IN LISTS _onnxruntime_search_paths)
        foreach(_suffix IN ITEMS lib lib64)
            if(EXISTS "${_dir}/${_suffix}")
                file(GLOB _candidates "${_dir}/${_suffix}/libonnxruntime*")
                if(_candidates)
                    list(GET _candidates 0 ONNXRuntime_LIBRARY)
                    break()
                endif()
            endif()
        endforeach()
        if(ONNXRuntime_LIBRARY)
            break()
        endif()
    endforeach()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS ONNXRuntime_LIBRARY ONNXRuntime_INCLUDE_DIR
)

if(ONNXRuntime_FOUND)
    set(ONNXRuntime_INCLUDE_DIRS ${ONNXRuntime_INCLUDE_DIR})
    set(ONNXRuntime_LIBRARIES ${ONNXRuntime_LIBRARY})

    if(NOT TARGET ONNXRuntime::ONNXRuntime)
        get_filename_component(_ort_ext "${ONNXRuntime_LIBRARY}" EXT)
        get_filename_component(_ort_dir "${ONNXRuntime_LIBRARY}" DIRECTORY)
        if(WIN32 AND _ort_ext STREQUAL ".lib")
            get_filename_component(_ort_name "${ONNXRuntime_LIBRARY}" NAME_WE)
            set(_ort_dll "${_ort_dir}/${_ort_name}.dll")
        else()
            set(_ort_dll "")
        endif()

        if(_ort_dll AND EXISTS "${_ort_dll}")
            # MSVC import library: the linker needs the .lib, the runtime the .dll
            add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED)
            set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
                IMPORTED_IMPLIB "${ONNXRuntime_LIBRARY}"
                IMPORTED_LOCATION "${_ort_dll}"
                INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}"
            )
        elseif(_ort_ext STREQUAL ".lib")
            # Import library without its DLL next to it: treat it as a plain library
            add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
            set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
                IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}"
            )
        else()
            add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED)
            set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
                IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}"
            )
        endif()
    endif()
endif()

mark_as_advanced(ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY)
