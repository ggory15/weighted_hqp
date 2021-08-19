/* 
 * This file has been automatically generated by the jrl-cmakemodules.
 * Please see https://github.com/jrl-umi3218/jrl-cmakemodules/blob/master/warning.hh.cmake for details.
*/

#ifndef HQP_HCOD_WARNING_HH
# define HQP_HCOD_WARNING_HH

// Emits a warning in a portable way.
//
// To emit a warning, one can insert:
//
// #pragma message HQP_HCOD_WARN("your warning message here")
//
// The use of this syntax is required as this is /not/ a standardized
// feature of C++ language or preprocessor, even if most of the
// compilers support it.

# define HQP_HCOD_WARN_STRINGISE_IMPL(x) #x
# define HQP_HCOD_WARN_STRINGISE(x) \
         HQP_HCOD_WARN_STRINGISE_IMPL(x)
# ifdef __GNUC__
#   define HQP_HCOD_WARN(exp) ("WARNING: " exp)
# else
#  ifdef _MSC_VER
#   define FILE_LINE_LINK __FILE__ "(" \
           HQP_HCOD_WARN_STRINGISE(__LINE__) ") : "
#   define HQP_HCOD_WARN(exp) (FILE_LINE_LINK "WARNING: " exp)
#  else
// If the compiler is not recognized, drop the feature.
#   define HQP_HCOD_WARN(MSG) /* nothing */
#  endif // __MSVC__
# endif // __GNUC__

#endif //! HQP_HCOD_WARNING_HH