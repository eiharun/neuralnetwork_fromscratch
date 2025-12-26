
#ifdef ENABLE_DEBUG
  #include <cstdio>
  #include <cstdlib>

  #if defined(__clang__) || defined(__GNUC__)
    #define LIB_TRAP() __builtin_trap()
  #elif defined(_MSC_VER)
    #define LIB_TRAP() __debugbreak()
  #else
    #define LIB_TRAP() abort()
  #endif

  #define LIB_ASSERT(cond, msg)                                \
    do {                                                       \
      if (!(cond)) {                                          \
        fprintf(stderr,                                      \
          "Assertion failed: %s:\t%s\n\tFile: %s\n\tLine: %d\n",    \
          #cond, msg, __FILE__, __LINE__);                     \
        LIB_TRAP();                                           \
      }                                                        \
    } while (0)
#else
  #define LIB_ASSERT(cond, msg) ((void)0)
#endif