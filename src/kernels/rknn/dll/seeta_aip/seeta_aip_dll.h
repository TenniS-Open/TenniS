//
// Created by Levalup on 2020/4/21.
//

#ifndef _INC_SEETA_AIP_DLL_H
#define _INC_SEETA_AIP_DLL_H

#include "seeta_aip_platform.h"
#include <string>
#include <sstream>
#include <iomanip>

#if SEETA_AIP_OS_UNIX || SEETA_AIP_OS_IOS

#include <dlfcn.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <fstream>

#define SEETA_AIP_GETCWD(buffer, length) ::getcwd((buffer), (length))
#define SEETA_AIP_CHDIR(path) ::chdir(path)

#elif SEETA_AIP_OS_WINDOWS

#include <direct.h>
#include <io.h>
#include <Windows.h>

#define SEETA_AIP_GETCWD(buffer, length) ::_getcwd((buffer), (length))
#define SEETA_AIP_CHDIR(path) ::_chdir(path)

/**
 * Undefined used variables in AIP
 */
#undef VOID
#undef min
#undef max
#undef TRUE
#undef FALSE
#undef BOOL

#else

#pragma message("[WRANING] Using system not support dynamic library loading!")

#endif

namespace seeta {
    namespace aip {
        /**
         * Load dynamic library
         * @param [in] libname full path to library name
         * @return handle to library
         * @note call dlclose to close handle
         * @note return null if failed. Call dlerror to get error message.
         */
        inline void *dlopen(const char *libname) {
#if SEETA_AIP_OS_UNIX
            // ::setenv("OMP_WAIT_POLICY", "passive", 1);
            auto handle = ::dlopen(libname, RTLD_LAZY | RTLD_LOCAL);
            return handle;
#elif SEETA_AIP_OS_WINDOWS
            ::SetEnvironmentVariableA("OMP_WAIT_POLICY", "passive");
            auto instance = ::LoadLibraryA(libname);
            return static_cast<void*>(instance);
#else
            (void)(libname);
            return nullptr;
#endif
        }

        /**
         * Open symbol in dynamic library
         * @param [in] handle return value of dlopen
         * @param [in] symbol symbol name
         * @return pointer to function
         */
        inline void *dlsym(void *handle, const char *symbol) {
#if SEETA_AIP_OS_UNIX
            return ::dlsym(handle, symbol);
#elif SEETA_AIP_OS_WINDOWS
            auto instance = static_cast<HMODULE>(handle);
            return reinterpret_cast<void*>(::GetProcAddress(instance, symbol));
#else
            (void)(handle);
            (void)(symbol);
            return nullptr;
#endif
        }

        /**
         * Close dynamic library
         * @param [in] handle return value of dlopen
         */
        inline void dlclose(void *handle) {
#if SEETA_AIP_OS_UNIX
            ::dlclose(handle);
#elif SEETA_AIP_OS_WINDOWS
            // ::SetEnvironmentVariableA("OMP_WAIT_POLICY", "passive");
            auto instance = static_cast<HMODULE>(handle);
            ::FreeLibrary(instance);
#else
            (void)(handle);
#endif
        }

#if SEETA_AIP_OS_WINDOWS
        inline std::string _format_message(DWORD dw) {
            LPVOID lpMsgBuf = nullptr;
            ::FormatMessageA(
                    FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
                    NULL,
                    dw,
                    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                    (LPTSTR) & lpMsgBuf,
                    0, NULL);
            std::ostringstream msg;
            msg << std::hex << "0x" << std::setw(8) << std::setfill('0') << dw << ": ";
            if (lpMsgBuf != nullptr) {
                msg << std::string(reinterpret_cast<char *>(lpMsgBuf));
                LocalFree(lpMsgBuf);
            }
            return msg.str();
        }
#endif

        /**
         * Get last dlopen error message
         * @return error message
         */
        inline std::string dlerror() {
#if SEETA_AIP_OS_UNIX
            auto error_message = ::dlerror();
            if (error_message) return error_message;
            return "No error.";
#elif SEETA_AIP_OS_WINDOWS
            return _format_message(::GetLastError());
#else
            return "dlopen undefined.";
#endif
        }

        inline std::string _cut_path_tail(const std::string &path, std::string &tail) {
            auto win_sep_pos = path.rfind('\\');
            auto unix_sep_pos = path.rfind('/');
            auto sep_pos = win_sep_pos;
            if (sep_pos == std::string::npos) sep_pos = unix_sep_pos;
            else if (unix_sep_pos != std::string::npos && unix_sep_pos > sep_pos) sep_pos = unix_sep_pos;
            if (sep_pos == std::string::npos) {
                tail = path;
                return std::string();
            }
            tail = path.substr(sep_pos + 1);
            return path.substr(0, sep_pos);
        }

        inline char _file_separator() {
#if SEETA_AIP_OS_WINDOWS
            return '\\';
#else
            return '/';
#endif
        }

        inline std::string _getcwd() {
            auto pwd = SEETA_AIP_GETCWD(nullptr, 0);
            if (pwd == nullptr) return std::string();
            std::string pwd_str = pwd;
            free(pwd);
            return pwd_str;
        }

        inline bool _chdir(const std::string &path) {
            return SEETA_AIP_CHDIR(path.c_str()) == 0;
        }


        class _stack_cd {
        public:
            using self = _stack_cd;

            _stack_cd(const _stack_cd &) = delete;

            _stack_cd &operator=(const _stack_cd &) = delete;

            explicit _stack_cd(const std::string &path)
            : m_cwd(_getcwd()) {
                _chdir(path);
            }

            ~_stack_cd() {
                _chdir(m_cwd);
            }

        private:
            std::string m_cwd;
        };

        inline void *_dlopen_v2(const std::string &root, const std::string &libname,
                                const std::string &prefix, const std::string &suffix) {
            std::ostringstream libpath;
            if (root.empty()) {
                libpath << prefix << libname << suffix;
            } else {
                libpath << root << _file_separator() << prefix << libname << suffix;
            }
            auto tmp = libpath.str();
            return dlopen(tmp.c_str());
        }

        /**
         * Load dynamic library
         * @param [in] libname path to libname, can ignore the the prefix or suffix of libname
         * @return handle to library
         * @note call dlclose to close handle
         * @note return null if failed. Call dlerror to get error message.
         * Example dlopen_v2("test") can load `test`, `libtest.so` or `test.so` on linux instead.
         */
        inline void *dlopen_v2(const std::string &libname) {
#if SEETA_AIP_OS_MAC || SEETA_AIP_OS_IOS
            static const char *prefix = "lib";
            static const char *suffix = ".dylib";
#elif SEETA_AIP_OS_UNIX
            static const char *prefix = "lib";
            static const char *suffix = ".so";
#elif SEETA_AIP_OS_WINDOWS
            static const char *prefix = "lib";
            static const char *suffix = ".dll";
#endif
            std::string tail;
            std::string head = _cut_path_tail(libname, tail);

            _stack_cd _cd_local(head);

            // first open library
            auto handle = dlopen(libname.c_str());
            if (handle) return handle;

            // try other names
            handle = _dlopen_v2(head, tail, prefix, suffix);
            if (handle) return handle;

            handle = _dlopen_v2(head, tail, "", suffix);
            if (handle) return handle;

            handle = _dlopen_v2(head, tail, prefix, "");
            if (handle) return handle;

            handle = _dlopen_v2("", tail, prefix, suffix);
            if (handle) return handle;

            handle = _dlopen_v2("", tail, "", suffix);
            if (handle) return handle;

            handle = _dlopen_v2("", tail, prefix, "");
            if (handle) return handle;

            handle = _dlopen_v2("", tail, "", "");
            if (handle) return handle;

            handle = dlopen(libname.c_str());
            if (handle) return handle;

            // final failed
            return nullptr;
        }
    }
}

#undef SEETA_AIP_CHDIR
#undef SEETA_AIP_GETCWD

#endif //_INC_SEETA_AIP_DLL_H
