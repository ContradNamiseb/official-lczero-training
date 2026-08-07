# Minimal meson build for zlib, used because WrapDB is unreachable from the
# development network. Upstream zlib ships autotools/CMake only.
#
# The loader itself needs just inflateInit2/inflate/inflateEnd, but the whole
# library is built so the CLI tools (dump_chunk, filter_chunks), which use the
# gzFile API, keep working.

project(
  'zlib',
  'c',
  version : '1.3.1',
  license : 'Zlib',
  default_options : ['warning_level=0'],
)

cc = meson.get_compiler('c')

zlib_c_args = []
if cc.get_id() == 'msvc'
  # zlib uses strcpy/sprintf and POSIX-named CRT functions throughout; MSVC
  # deprecates both and would otherwise bury the build in warnings.
  zlib_c_args += [
    '-D_CRT_SECURE_NO_DEPRECATE',
    '-D_CRT_NONSTDC_NO_DEPRECATE',
  ]
else
  zlib_c_args += ['-DHAVE_UNISTD_H']
endif

# Suppress the handful of warnings zlib emits under any compiler; they are
# upstream's business, not this project's.
zlib_c_args += cc.get_supported_arguments([
  '-Wno-implicit-function-declaration',
  '-Wno-deprecated-non-prototype',
])

zlib_sources = files(
  'adler32.c',
  'compress.c',
  'crc32.c',
  'deflate.c',
  'gzclose.c',
  'gzlib.c',
  'gzread.c',
  'gzwrite.c',
  'infback.c',
  'inffast.c',
  'inflate.c',
  'inftrees.c',
  'trees.c',
  'uncompr.c',
  'zutil.c',
)

zlib_inc = include_directories('.')

zlib_lib = static_library(
  'z',
  zlib_sources,
  c_args : zlib_c_args,
  include_directories : zlib_inc,
)

zlib_dep = declare_dependency(
  link_with : zlib_lib,
  include_directories : zlib_inc,
)

meson.override_dependency('zlib', zlib_dep)
