/*
 * libembedding - win_psapi_init.hpp
 * Windows PSAPI initialization
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */
/*
 * libembedding - detail/win_psapi_init.hpp
 * Force include of psapi.h BEFORE curl and other Windows headers
 * to avoid "DWORD redefinition" and "LPVOID undeclared" errors.
 *
 * SPDX-License-Identifier: MIT
 */

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <psapi.h>
#  include <winnt.h>
#endif

