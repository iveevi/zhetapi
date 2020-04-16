/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with f or F, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define FBISON 1

/* Bison version.  */
#define FBISON_VERSION "3.0.4"

/* Skeleton name.  */
#define FSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define FPURE 0

/* Push parsers.  */
#define FPUSH 0

/* Pull parsers.  */
#define FPULL 1

/* Substitute the type names.
#define F	F
#define f	f
#define FSTYPE         FSTYPE
* Substitute the variable and function names.  *
* #define fparse         fparse
#define flex           flex
#define ferror         ferror
#define fdebug         fdebug
#define fnerrs         fnerrs

#define flval          flval
#define fchar          fchar */

/* Copy the first part of user declarations.  */
#line 1 "function_parser.y" /* yacc.c:339  */

	#include <iostream>
	#include <cctype>
	#include <cstring>
	#include <vector>
	#include <stack>
	#include <string>

	#include "operation.h"
	#include "defaults.h"
  	#include "operand.h"

	#include "functor.h"
	#include "func_stack.h"

	#include "var_stack.h"
	#include "variable.h"

	extern "C" int flex();
  	
	using namespace std;

	void ferror(functor <double> ::node *(&), functor <double> ::param_list, functor <double> ::map, const char *);

#line 100 "function_parser.h" /* yacc.c:339  */

# ifndef F_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define F_NULLPTR nullptr
#  else
#   define F_NULLPTR 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef FERROR_VERBOSE
# undef FERROR_VERBOSE
# define FERROR_VERBOSE 1
#else
# define FERROR_VERBOSE 1
#endif


/* Debug traces.  */
#ifndef FDEBUG
# if defined FDEBUG
#if FDEBUG
#   define FDEBUG 1
#  else
#   define FDEBUG 0
#  endif
# else /* ! defined FDEBUG */
#  define FDEBUG 0
# endif /* ! defined FDEBUG */
#endif  /* ! defined FDEBUG */
#if FDEBUG
extern int fdebug;
#endif

/* Token type.  */
#ifndef FTOKENTYPE
# define FTOKENTYPE
  enum ftokentype
  {
    F_IDENT = 258,
    F_NUMBER = 259,
    F_PLUS = 260,
    F_MINUS = 261,
    F_MULT = 262,
    F_DIV = 263,
    F_SIN = 264,
    F_COS = 265,
    F_TAN = 266,
    F_CSC = 267,
    F_SEC = 268,
    F_COT = 269,
    F_LOG = 270,
    F_LN = 271,
    F_LG = 272,
    F_SUPERSCRIPT = 273,
    F_SUBSCRIPT = 274,
    F_LPAREN = 275,
    F_RPAREN = 276,
    F_LBRACE = 277,
    F_RBRACE = 278,
    F_LBRACKET = 279,
    F_RBRACKET = 280,
    F_END = 281
  };
#endif

/* Value type.  */
#if ! defined FSTYPE && ! defined FSTYPE_IS_DECLARED

union FSTYPE
{
#line 55 "function_parser.y" /* yacc.c:355  */

	functor <double> ::node		*expr;
	functor <double> ::node		*coll;
	functor <double> ::node		*term;
	functor <double> ::node		*felm;
	functor <double> ::node		*dopn;
	functor <double> ::node		*dpnt;
	functor <double> ::node		*prth;
	functor <double> ::node		*sclr;

	const char			*ident;
	double       			value;

#line 186 "function_parser.h" /* yacc.c:355  */
};

typedef union FSTYPE FSTYPE;
# define FSTYPE_IS_TRIVIAL 1
# define FSTYPE_IS_DECLARED 1
#endif


extern FSTYPE flval;

int fparse (functor <double> ::node *(&root), functor <double> ::param_list list, functor <double> ::map &vmap);



/* Copy the second part of user declarations.  */

#line 203 "function_parser.h" /* yacc.c:358  */

#ifdef short
# undef short
#endif

#ifdef FTYPE_UINT8
typedef FTYPE_UINT8 ftype_uint8;
#else
typedef unsigned char ftype_uint8;
#endif

#ifdef FTYPE_INT8
typedef FTYPE_INT8 ftype_int8;
#else
typedef signed char ftype_int8;
#endif

#ifdef FTYPE_UINT16
typedef FTYPE_UINT16 ftype_uint16;
#else
typedef unsigned short int ftype_uint16;
#endif

#ifdef FTYPE_INT16
typedef FTYPE_INT16 ftype_int16;
#else
typedef short int ftype_int16;
#endif

#ifndef FSIZE_T
# ifdef __SIZE_TYPE__
#  define FSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define FSIZE_T size_t
# elif ! defined FSIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define FSIZE_T size_t
# else
#  define FSIZE_T unsigned int
# endif
#endif

#define FSIZE_MAXIMUM ((FSIZE_T) -1)

#ifndef F_
# if defined FENABLE_NLS && FENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define F_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef F_
#  define F_(Msgid) Msgid
# endif
#endif

#ifndef F_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define F_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define F_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef F_ATTRIBUTE_PURE
# define F_ATTRIBUTE_PURE   F_ATTRIBUTE ((__pure__))
#endif

#ifndef F_ATTRIBUTE_UNUSED
# define F_ATTRIBUTE_UNUSED F_ATTRIBUTE ((__unused__))
#endif

#if !defined _Noreturn \
     && (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112)
# if defined _MSC_VER && 1200 <= _MSC_VER
#  define _Noreturn __declspec (noreturn)
# else
#  define _Noreturn F_ATTRIBUTE ((__noreturn__))
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define FUSE(E) ((void) (E))
#else
# define FUSE(E) /* empty */
#endif

#if defined __GNUC__ && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about flval being uninitialized.  */
# define F_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define F_IGNORE_MAYBE_UNINITIALIZED_END \
    _Pragma ("GCC diagnostic pop")
#else
# define F_INITIAL_VALUE(Value) Value
#endif
#ifndef F_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define F_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define F_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef F_INITIAL_VALUE
# define F_INITIAL_VALUE(Value) /* Nothing. */
#endif


#if ! defined foverflow || FERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef FSTACK_USE_ALLOCA
#  if FSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define FSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define FSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define FSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef FSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define FSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef FSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define FSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define FSTACK_ALLOC FMALLOC
#  define FSTACK_FREE FFREE
#  ifndef FSTACK_ALLOC_MAXIMUM
#   define FSTACK_ALLOC_MAXIMUM FSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined FMALLOC || defined malloc) \
             && (defined FFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef FMALLOC
#   define FMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (FSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef FFREE
#   define FFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined foverflow || FERROR_VERBOSE */


#if (! defined foverflow \
     && (! defined __cplusplus \
         || (defined FSTYPE_IS_TRIVIAL && FSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union falloc
{
  ftype_int16 fss_alloc;
  FSTYPE fvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define FSTACK_GAP_MAXIMUM (sizeof (union falloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define FSTACK_BYTES(N) \
     ((N) * (sizeof (ftype_int16) + sizeof (FSTYPE)) \
      + FSTACK_GAP_MAXIMUM)

# define FCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables FSIZE and FSTACKSIZE give the old and new number of
   elements in the stack, and FPTR gives the new location of the
   stack.  Advance FPTR to a properly aligned location for the next
   stack.  */
# define FSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        FSIZE_T fnewbytes;                                            \
        FCOPY (&fptr->Stack_alloc, Stack, fsize);                    \
        Stack = &fptr->Stack_alloc;                                    \
        fnewbytes = fstacksize * sizeof (*Stack) + FSTACK_GAP_MAXIMUM; \
        fptr += fnewbytes / sizeof (*fptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined FCOPY_NEEDED && FCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef FCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define FCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define FCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          FSIZE_T fi;                         \
          for (fi = 0; fi < (Count); fi++)   \
            (Dst)[fi] = (Src)[fi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !FCOPY_NEEDED */

/* FFINAL -- State number of the termination state.  */
#define FFINAL  35
/* FLAST -- Last index in FTABLE.  */
#define FLAST   112

/* FNTOKENS -- Number of terminals.  */
#define FNTOKENS  27
/* FNNTS -- Number of nonterminals.  */
#define FNNTS  10
/* FNRULES -- Number of rules.  */
#define FNRULES  31
/* FNSTATES -- Number of states.  */
#define FNSTATES  56

/* FTRANSLATE[FX] -- Symbol number corresponding to FX as returned
   by flex, with out-of-bounds checking.  */
#define FUNDEFTOK  2
#define FMAXUTOK   281

#define FTRANSLATE(FX)                                                \
  ((unsigned int) (FX) <= FMAXUTOK ? ftranslate[FX] : FUNDEFTOK)

/* FTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by flex, without out-of-bounds checking.  */
static const ftype_uint8 ftranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26
};

#if FDEBUG
  /* FRLINE[FN] -- Source line where rule number FN was defined.  */
static const ftype_uint16 frline[] =
{
       0,   101,   101,   112,   126,   140,   154,   168,   182,   197,
     203,   217,   221,   228,   237,   243,   259,   276,   298,   319,
     331,   343,   355,   367,   379,   392,   407,   411,   415,   420,
     491,   499
};
#endif

#if FDEBUG || FERROR_VERBOSE || 1
/* FTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at FNTOKENS, nonterminals.  */
static const char *const ftname[] =
{
  "$end", "error", "$undefined", "F_IDENT", "F_NUMBER", "F_PLUS",
  "F_MINUS", "F_MULT", "F_DIV", "F_SIN", "F_COS", "F_TAN", "F_CSC",
  "F_SEC", "F_COT", "F_LOG", "F_LN", "F_LG", "F_SUPERSCRIPT",
  "F_SUBSCRIPT", "F_LPAREN", "F_RPAREN", "F_LBRACE", "F_RBRACE",
  "F_LBRACKET", "F_RBRACKET", "F_END", "$accept", "input", "expr", "coll",
  "term", "felm", "dopn", "dpnt", "sclr", "prth", F_NULLPTR
};
#endif

# ifdef FPRINT
/* FTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const ftype_uint16 ftoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281
};
# endif

#define FPACT_NINF -20

#define fpact_value_is_default(Yystate) \
  (!!((Yystate) == (-20)))

#define FTABLE_NINF -1

#define ftable_value_is_error(Yytable_value) \
  0

  /* FPACT[STATE-NUM] -- Index in FTABLE of the portion describing
     STATE-NUM.  */
static const ftype_int8 fpact[] =
{
      60,   -20,   -20,    75,    60,    60,    60,    60,    60,    60,
      42,    60,    60,    60,    22,    12,   -20,    75,   -20,     3,
     -20,   -20,   -20,   -20,   -20,   -20,   -20,   -20,   -20,   -20,
       1,   -20,   -20,   -20,    91,   -20,    60,    60,    60,    60,
      60,   -20,     7,   -20,     7,    60,   -20,    21,    21,     6,
       6,     6,     3,     8,    60,   -20
};

  /* FDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when FTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const ftype_uint8 fdefact[] =
{
       0,    29,    30,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     9,    12,    11,    14,
      26,    27,    28,     8,    24,    23,    22,    21,    20,    19,
       0,    18,    17,    16,     0,     1,     0,     0,     0,     0,
       0,     2,    13,    10,     0,     0,    31,     6,     7,     4,
       5,     3,    25,     0,     0,    15
};

  /* FPGOTO[NTERM-NUM].  */
static const ftype_int8 fpgoto[] =
{
     -20,   -20,    -4,    37,    -5,    25,   -19,   -20,   -20,   -20
};

  /* FDEFGOTO[NTERM-NUM].  */
static const ftype_int8 fdefgoto[] =
{
      -1,    14,    15,    16,    17,    18,    19,    20,    21,    22
};

  /* FTABLE[FPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If FTABLE_NINF, syntax error.  */
static const ftype_uint8 ftable[] =
{
      24,    25,    26,    27,    28,    29,    31,    32,    33,    34,
       1,     2,    42,    36,    37,    38,    39,    36,    37,    38,
      39,    44,    35,    45,    40,    52,    40,    13,    38,    39,
      40,    54,    47,    48,    49,    50,    51,    42,    41,    40,
      23,    53,    43,     0,     0,     1,     2,     0,     3,     0,
      55,     4,     5,     6,     7,     8,     9,    10,    11,    12,
       0,    30,    13,     1,     2,     0,     3,     0,     0,     4,
       5,     6,     7,     8,     9,    10,    11,    12,     1,     2,
      13,     0,     0,     0,     4,     5,     6,     7,     8,     9,
      10,    11,    12,     0,     0,    13,    36,    37,    38,    39,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    40,
       0,     0,    46
};

static const ftype_int8 fcheck[] =
{
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
       3,     4,    17,     5,     6,     7,     8,     5,     6,     7,
       8,    18,     0,    22,    18,    44,    18,    20,     7,     8,
      18,    23,    36,    37,    38,    39,    40,    42,    26,    18,
       3,    45,    17,    -1,    -1,     3,     4,    -1,     6,    -1,
      54,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      -1,    19,    20,     3,     4,    -1,     6,    -1,    -1,     9,
      10,    11,    12,    13,    14,    15,    16,    17,     3,     4,
      20,    -1,    -1,    -1,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    -1,    -1,    20,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    18,
      -1,    -1,    21
};

  /* FSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const ftype_uint8 fstos[] =
{
       0,     3,     4,     6,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    20,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    30,    29,    29,    29,    29,    29,    29,
      19,    29,    29,    29,    29,     0,     5,     6,     7,     8,
      18,    26,    31,    32,    18,    22,    21,    29,    29,    29,
      29,    29,    33,    29,    23,    29
};

  /* FR1[FN] -- Symbol number of symbol that rule FN derives.  */
static const ftype_uint8 fr1[] =
{
       0,    27,    28,    29,    29,    29,    29,    29,    29,    29,
      30,    30,    30,    31,    31,    32,    32,    32,    32,    32,
      32,    32,    32,    32,    32,    33,    33,    33,    33,    34,
      35,    36
};

  /* FR2[FN] -- Number of symbols on the right hand side of rule FN.  */
static const ftype_uint8 fr2[] =
{
       0,     2,     2,     3,     3,     3,     3,     3,     2,     1,
       2,     1,     1,     2,     1,     6,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     3,     1,     1,     1,     1,
       1,     3
};


#define ferrok         (ferrstatus = 0)
#define fclearin       (fchar = FEMPTY)
#define FEMPTY         (-2)
#define FEOF           0

#define FACCEPT        goto facceptlab
#define FABORT         goto fabortlab
#define FERROR         goto ferrorlab


#define FRECOVERING()  (!!ferrstatus)

#define FBACKUP(Token, Value)                                  \
do                                                              \
  if (fchar == FEMPTY)                                        \
    {                                                           \
      fchar = (Token);                                         \
      flval = (Value);                                         \
      FPOPSTACK (flen);                                       \
      fstate = *fssp;                                         \
      goto fbackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      ferror (root, list, vmap, F_("syntax error: cannot back up")); \
      FERROR;                                                  \
    }                                                           \
while (0)

/* Error token number */
#define FTERROR        1
#define FERRCODE       256



/* Enable debugging if requested.  */
#if FDEBUG

# ifndef FFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define FFPRINTF fprintf
# endif

# define FDPRINTF(Args)                        \
do {                                            \
  if (fdebug)                                  \
    FFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
#ifndef F_LOCATION_PRINT
# define F_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


# define F_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (fdebug)                                                            \
    {                                                                     \
      FFPRINTF (stderr, "%s ", Title);                                   \
      f_symbol_print (stderr,                                            \
                  Type, Value, root, list, vmap); \
      FFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*----------------------------------------.
| Print this symbol's value on FOUTPUT.  |
`----------------------------------------*/

static void
f_symbol_value_print (FILE *foutput, int ftype, FSTYPE const * const fvaluep, functor <double> ::node *(&root), functor <double> ::param_list list, functor <double> ::map &vmap)
{
  FILE *fo = foutput;
  FUSE (fo);
  FUSE (root);
  FUSE (list);
  FUSE (vmap);
  if (!fvaluep)
    return;
# ifdef FPRINT
  if (ftype < FNTOKENS)
    FPRINT (foutput, ftoknum[ftype], *fvaluep);
# endif
  FUSE (ftype);
}


/*--------------------------------.
| Print this symbol on FOUTPUT.  |
`--------------------------------*/

static void
f_symbol_print (FILE *foutput, int ftype, FSTYPE const * const fvaluep, functor <double> ::node *(&root), functor <double> ::param_list list, functor <double> ::map &vmap)
{
  FFPRINTF (foutput, "%s %s (",
             ftype < FNTOKENS ? "token" : "nterm", ftname[ftype]);

  f_symbol_value_print (foutput, ftype, fvaluep, root, list, vmap);
  FFPRINTF (foutput, ")");
}

/*------------------------------------------------------------------.
| f_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
f_stack_print (ftype_int16 *fbottom, ftype_int16 *ftop)
{
  FFPRINTF (stderr, "Stack now");
  for (; fbottom <= ftop; fbottom++)
    {
      int fbot = *fbottom;
      FFPRINTF (stderr, " %d", fbot);
    }
  FFPRINTF (stderr, "\n");
}

# define F_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (fdebug)                                                  \
    f_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the FRULE is going to be reduced.  |
`------------------------------------------------*/

static void
f_reduce_print (ftype_int16 *fssp, FSTYPE *fvsp, int frule, functor <double> ::node *(&root), functor <double> ::param_list list, functor <double> ::map &vmap)
{
  unsigned long int flno = frline[frule];
  int fnrhs = fr2[frule];
  int fi;
  FFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             frule - 1, flno);
  /* The symbols being reduced.  */
  for (fi = 0; fi < fnrhs; fi++)
    {
      FFPRINTF (stderr, "   $%d = ", fi + 1);
      f_symbol_print (stderr,
                       fstos[fssp[fi + 1 - fnrhs]],
                       &(fvsp[(fi + 1) - (fnrhs)])
                                              , root, list, vmap);
      FFPRINTF (stderr, "\n");
    }
}

# define F_REDUCE_PRINT(Rule)          \
do {                                    \
  if (fdebug)                          \
    f_reduce_print (fssp, fvsp, Rule, root, list, vmap); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int fdebug;
#else /* !FDEBUG */
# define FDPRINTF(Args)
# define F_SYMBOL_PRINT(Title, Type, Value, Location)
# define F_STACK_PRINT(Bottom, Top)
# define F_REDUCE_PRINT(Rule)
#endif /* !FDEBUG */


/* FINITDEPTH -- initial size of the parser's stacks.  */
#ifndef FINITDEPTH
# define FINITDEPTH 200
#endif

/* FMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   FSTACK_ALLOC_MAXIMUM < FSTACK_BYTES (FMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef FMAXDEPTH
# define FMAXDEPTH 10000
#endif


#if FERROR_VERBOSE

# ifndef fstrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define fstrlen strlen
#  else
/* Return the length of FSTR.  */
static FSIZE_T
fstrlen (const char *fstr)
{
  FSIZE_T flen;
  for (flen = 0; fstr[flen]; flen++)
    continue;
  return flen;
}
#  endif
# endif

# ifndef fstpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define fstpcpy stpcpy
#  else
/* Copy FSRC to FDEST, returning the address of the terminating '\0' in
   FDEST.  */
static char *
fstpcpy (char *fdest, const char *fsrc)
{
  char *fd = fdest;
  const char *fs = fsrc;

  while ((*fd++ = *fs++) != '\0')
    continue;

  return fd - 1;
}
#  endif
# endif

# ifndef ftnamerr
/* Copy to FRES the contents of FSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for ferror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  FSTR is taken from ftname.  If FRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static FSIZE_T
ftnamerr (char *fres, const char *fstr)
{
  if (*fstr == '"')
    {
      FSIZE_T fn = 0;
      char const *fp = fstr;

      for (;;)
        switch (*++fp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++fp != '\\')
              goto do_not_strip_quotes;
            /* Fall through.  */
          default:
            if (fres)
              fres[fn] = *fp;
            fn++;
            break;

          case '"':
            if (fres)
              fres[fn] = '\0';
            return fn;
          }
    do_not_strip_quotes: ;
    }

  if (! fres)
    return fstrlen (fstr);

  return fstpcpy (fres, fstr) - fres;
}
# endif

/* Copy into *FMSG, which is of size *FMSG_ALLOC, an error message
   about the unexpected token FTOKEN for the state stack whose top is
   FSSP.

   Return 0 if *FMSG was successfully written.  Return 1 if *FMSG is
   not large enough to hold the message.  In that case, also set
   *FMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
fsyntax_error (FSIZE_T *fmsg_alloc, char **fmsg,
                ftype_int16 *fssp, int ftoken)
{
  FSIZE_T fsize0 = ftnamerr (F_NULLPTR, ftname[ftoken]);
  FSIZE_T fsize = fsize0;
  enum { FERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *fformat = F_NULLPTR;
  /* Arguments of fformat. */
  char const *farg[FERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int fcount = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in fchar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated fchar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (ftoken != FEMPTY)
    {
      int fn = fpact[*fssp];
      farg[fcount++] = ftname[ftoken];
      if (!fpact_value_is_default (fn))
        {
          /* Start FX at -FN if negative to avoid negative indexes in
             FCHECK.  In other words, skip the first -FN actions for
             this state because they are default actions.  */
          int fxbegin = fn < 0 ? -fn : 0;
          /* Stay within bounds of both fcheck and ftname.  */
          int fchecklim = FLAST - fn + 1;
          int fxend = fchecklim < FNTOKENS ? fchecklim : FNTOKENS;
          int fx;

          for (fx = fxbegin; fx < fxend; ++fx)
            if (fcheck[fx + fn] == fx && fx != FTERROR
                && !ftable_value_is_error (ftable[fx + fn]))
              {
                if (fcount == FERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    fcount = 1;
                    fsize = fsize0;
                    break;
                  }
                farg[fcount++] = ftname[fx];
                {
                  FSIZE_T fsize1 = fsize + ftnamerr (F_NULLPTR, ftname[fx]);
                  if (! (fsize <= fsize1
                         && fsize1 <= FSTACK_ALLOC_MAXIMUM))
                    return 2;
                  fsize = fsize1;
                }
              }
        }
    }

  switch (fcount)
    {
# define FCASE_(N, S)                      \
      case N:                               \
        fformat = S;                       \
      break
      FCASE_(0, F_("syntax error"));
      FCASE_(1, F_("syntax error, unexpected %s"));
      FCASE_(2, F_("syntax error, unexpected %s, expecting %s"));
      FCASE_(3, F_("syntax error, unexpected %s, expecting %s or %s"));
      FCASE_(4, F_("syntax error, unexpected %s, expecting %s or %s or %s"));
      FCASE_(5, F_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef FCASE_
    }

  {
    FSIZE_T fsize1 = fsize + fstrlen (fformat);
    if (! (fsize <= fsize1 && fsize1 <= FSTACK_ALLOC_MAXIMUM))
      return 2;
    fsize = fsize1;
  }

  if (*fmsg_alloc < fsize)
    {
      *fmsg_alloc = 2 * fsize;
      if (! (fsize <= *fmsg_alloc
             && *fmsg_alloc <= FSTACK_ALLOC_MAXIMUM))
        *fmsg_alloc = FSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *fp = *fmsg;
    int fi = 0;
    while ((*fp = *fformat) != '\0')
      if (*fp == '%' && fformat[1] == 's' && fi < fcount)
        {
          fp += ftnamerr (fp, farg[fi++]);
          fformat += 2;
        }
      else
        {
          fp++;
          fformat++;
        }
  }
  return 0;
}
#endif /* FERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
fdestruct (const char *fmsg, int ftype, FSTYPE *fvaluep, functor <double> ::node *(&root), functor <double> ::param_list list, functor <double> ::map &vmap)
{
  FUSE (fvaluep);
  FUSE (root);
  FUSE (list);
  FUSE (vmap);
  if (!fmsg)
    fmsg = "Deleting";
  F_SYMBOL_PRINT (fmsg, ftype, fvaluep, flocationp);

  F_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  FUSE (ftype);
  F_IGNORE_MAYBE_UNINITIALIZED_END
}




/* The lookahead symbol.  */
int fchar;

/* The semantic value of the lookahead symbol.  */
FSTYPE flval;
/* Number of syntax errors so far.  */
int fnerrs;


/*----------.
| fparse.  |
`----------*/

int
fparse (functor <double> ::node *(&root), functor <double> ::param_list list, functor <double> ::map &vmap)
{
    int fstate;
    /* Number of tokens to shift before error messages enabled.  */
    int ferrstatus;

    /* The stacks and their tools:
       'fss': related to states.
       'fvs': related to semantic values.

       Refer to the stacks through separate pointers, to allow foverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    ftype_int16 fssa[FINITDEPTH];
    ftype_int16 *fss;
    ftype_int16 *fssp;

    /* The semantic value stack.  */
    FSTYPE fvsa[FINITDEPTH];
    FSTYPE *fvs;
    FSTYPE *fvsp;

    FSIZE_T fstacksize;

  int fn;
  int fresult;
  /* Lookahead token as an internal (translated) token number.  */
  int ftoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  FSTYPE fval;

#if FERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char fmsgbuf[128];
  char *fmsg = fmsgbuf;
  FSIZE_T fmsg_alloc = sizeof fmsgbuf;
#endif

#define FPOPSTACK(N)   (fvsp -= (N), fssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int flen = 0;

  fssp = fss = fssa;
  fvsp = fvs = fvsa;
  fstacksize = FINITDEPTH;

  FDPRINTF ((stderr, "Starting parse\n"));

  fstate = 0;
  ferrstatus = 0;
  fnerrs = 0;
  fchar = FEMPTY; /* Cause a token to be read.  */
  goto fsetstate;

/*------------------------------------------------------------.
| fnewstate -- Push a new state, which is found in fstate.  |
`------------------------------------------------------------*/
 fnewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  fssp++;

 fsetstate:
  *fssp = fstate;

  if (fss + fstacksize - 1 <= fssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      FSIZE_T fsize = fssp - fss + 1;

#ifdef foverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        FSTYPE *fvs1 = fvs;
        ftype_int16 *fss1 = fss;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if foverflow is a macro.  */
        foverflow (F_("memory exhausted"),
                    &fss1, fsize * sizeof (*fssp),
                    &fvs1, fsize * sizeof (*fvsp),
                    &fstacksize);

        fss = fss1;
        fvs = fvs1;
      }
#else /* no foverflow */
# ifndef FSTACK_RELOCATE
      goto fexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (FMAXDEPTH <= fstacksize)
        goto fexhaustedlab;
      fstacksize *= 2;
      if (FMAXDEPTH < fstacksize)
        fstacksize = FMAXDEPTH;

      {
        ftype_int16 *fss1 = fss;
        union falloc *fptr =
          (union falloc *) FSTACK_ALLOC (FSTACK_BYTES (fstacksize));
        if (! fptr)
          goto fexhaustedlab;
        FSTACK_RELOCATE (fss_alloc, fss);
        FSTACK_RELOCATE (fvs_alloc, fvs);
#  undef FSTACK_RELOCATE
        if (fss1 != fssa)
          FSTACK_FREE (fss1);
      }
# endif
#endif /* no foverflow */

      fssp = fss + fsize - 1;
      fvsp = fvs + fsize - 1;

      FDPRINTF ((stderr, "Stack size increased to %lu\n",
                  (unsigned long int) fstacksize));

      if (fss + fstacksize - 1 <= fssp)
        FABORT;
    }

  FDPRINTF ((stderr, "Entering state %d\n", fstate));

  if (fstate == FFINAL)
    FACCEPT;

  goto fbackup;

/*-----------.
| fbackup.  |
`-----------*/
fbackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  fn = fpact[fstate];
  if (fpact_value_is_default (fn))
    goto fdefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* FCHAR is either FEMPTY or FEOF or a valid lookahead symbol.  */
  if (fchar == FEMPTY)
    {
      FDPRINTF ((stderr, "Reading a token: "));
      fchar = flex ();
    }

  if (fchar <= FEOF)
    {
      fchar = ftoken = FEOF;
      FDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      ftoken = FTRANSLATE (fchar);
      F_SYMBOL_PRINT ("Next token is", ftoken, &flval, &flloc);
    }

  /* If the proper action on seeing token FTOKEN is to reduce or to
     detect an error, take that action.  */
  fn += ftoken;
  if (fn < 0 || FLAST < fn || fcheck[fn] != ftoken)
    goto fdefault;
  fn = ftable[fn];
  if (fn <= 0)
    {
      if (ftable_value_is_error (fn))
        goto ferrlab;
      fn = -fn;
      goto freduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (ferrstatus)
    ferrstatus--;

  /* Shift the lookahead token.  */
  F_SYMBOL_PRINT ("Shifting", ftoken, &flval, &flloc);

  /* Discard the shifted token.  */
  fchar = FEMPTY;

  fstate = fn;
  F_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++fvsp = flval;
  F_IGNORE_MAYBE_UNINITIALIZED_END

  goto fnewstate;


/*-----------------------------------------------------------.
| fdefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
fdefault:
  fn = fdefact[fstate];
  if (fn == 0)
    goto ferrlab;
  goto freduce;


/*-----------------------------.
| freduce -- Do a reduction.  |
`-----------------------------*/
freduce:
  /* fn is the number of a rule to reduce with.  */
  flen = fr2[fn];

  /* If FLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets FVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to FVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that FVAL may be used uninitialized.  */
  fval = fvsp[1-flen];


  F_REDUCE_PRINT (fn);
  switch (fn)
    {
        case 2:
#line 101 "function_parser.y" /* yacc.c:1646  */
    {
     		// value->set($1->get());
		//printf("end of input, root is %s\n", $1->tok->str().c_str());
		root = (fvsp[-1].expr);
		// root = new functor <double> ::node {new operand <double> (10), functor <double> ::m_none, vector <functor <double> ::node *> ()};
		// functor <double> ::print($1, 1, 0);
		// functor <double> ::print(root, 1, 0);
		return 0;
}
#line 1337 "function_parser.h" /* yacc.c:1646  */
    break;

  case 3:
#line 112 "function_parser.y" /* yacc.c:1646  */
    { // Exponentiation
   		//printf("expression exponentiation\n");
		/* vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::exp_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back((fvsp[-2].expr));
		leaves.push_back((fvsp[0].expr));

		(fval.expr) = new functor <double> ::node {&defaults <double> ::exp_op, functor <double> ::m_none, leaves};
}
#line 1355 "function_parser.h" /* yacc.c:1646  */
    break;

  case 4:
#line 126 "function_parser.y" /* yacc.c:1646  */
    { // Multiplication
   		//printf("expression multiplication\n");
		/* vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::mult_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back((fvsp[-2].expr));
		leaves.push_back((fvsp[0].expr));

		(fval.expr) = new functor <double> ::node {&defaults <double> ::mult_op, functor <double> ::m_none, leaves};
}
#line 1373 "function_parser.h" /* yacc.c:1646  */
    break;

  case 5:
#line 140 "function_parser.y" /* yacc.c:1646  */
    { // Division
   		//printf("expression divition\n");
		/* tor <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::div_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back((fvsp[-2].expr));
		leaves.push_back((fvsp[0].expr));

		(fval.expr) = new functor <double> ::node {&defaults <double> ::div_op, functor <double> ::m_none, leaves};
}
#line 1391 "function_parser.h" /* yacc.c:1646  */
    break;

  case 6:
#line 154 "function_parser.y" /* yacc.c:1646  */
    { // Addition
   		//printf("expression addition\n");
		/* vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::add_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back((fvsp[-2].expr));
		leaves.push_back((fvsp[0].expr));

		(fval.expr) = new functor <double> ::node {&defaults <double> ::add_op, functor <double> ::m_none, leaves};
}
#line 1409 "function_parser.h" /* yacc.c:1646  */
    break;

  case 7:
#line 168 "function_parser.y" /* yacc.c:1646  */
    { // Subtraction
   		//printf("expression substraction\n");
   		/* vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::sub_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back((fvsp[-2].expr));
		leaves.push_back((fvsp[0].expr));

		(fval.expr) = new functor <double> ::node {&defaults <double> ::sub_op, functor <double> ::m_none, leaves};
}
#line 1427 "function_parser.h" /* yacc.c:1646  */
    break;

  case 8:
#line 182 "function_parser.y" /* yacc.c:1646  */
    {
   		//printf("expression negative collective\n");
   		/* vector <operand <double>> vals;
		vals.push_back(operand <double> (-1));
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::mult_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back(new functor <double> ::node (new operand <double> (-1), functor <double> ::m_none,
			vector <functor <double> ::node *> ()));
		leaves.push_back((fvsp[0].coll));

		(fval.expr) = new functor <double> ::node {&defaults <double> ::sub_op, functor <double> ::m_none, leaves};
}
#line 1446 "function_parser.h" /* yacc.c:1646  */
    break;

  case 9:
#line 197 "function_parser.y" /* yacc.c:1646  */
    {
   		//printf("expression collective\n");
   		(fval.expr) = (fvsp[0].coll);
}
#line 1455 "function_parser.h" /* yacc.c:1646  */
    break;

  case 10:
#line 203 "function_parser.y" /* yacc.c:1646  */
    { // Implicit Multiplication: term and non-arithmetic operation
    		//printf("collective, term (%s) implicitly multiplicied with non-arithmetic operation (%s)\n", $1->tok->str().c_str(), $2->tok->str().c_str());
		/* vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::mult_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back((fvsp[-1].term));
		leaves.push_back((fvsp[0].felm));

		(fval.coll) = new functor <double> ::node (&defaults <double> ::mult_op, functor <double> ::m_none, leaves);
}
#line 1473 "function_parser.h" /* yacc.c:1646  */
    break;

  case 11:
#line 217 "function_parser.y" /* yacc.c:1646  */
    {
    		(fval.coll) = (fvsp[0].felm);
}
#line 1481 "function_parser.h" /* yacc.c:1646  */
    break;

  case 12:
#line 221 "function_parser.y" /* yacc.c:1646  */
    {

    		// printf("collective as a regular term (%s)\n", $1->str().c_str());
    		(fval.coll) = (fvsp[0].term);
}
#line 1491 "function_parser.h" /* yacc.c:1646  */
    break;

  case 13:
#line 228 "function_parser.y" /* yacc.c:1646  */
    { // Implicit Multiplication: two or more terms
    		//printf("term with two terms, %s and %s\n", $1->tok->str().c_str(), $2->tok->str().c_str());
		vector <functor <double> ::node *> leaves;
		leaves.push_back((fvsp[-1].term));
		leaves.push_back((fvsp[0].term));

		(fval.term) = new functor <double> ::node (&defaults <double> ::mult_op, functor <double> ::m_none, leaves);
}
#line 1504 "function_parser.h" /* yacc.c:1646  */
    break;

  case 14:
#line 237 "function_parser.y" /* yacc.c:1646  */
    { // Direct Operand
    		//printf("term with direct operand %s\n", $1->tok->str().c_str());
    		(fval.term) = (fvsp[0].dopn);
}
#line 1513 "function_parser.h" /* yacc.c:1646  */
    break;

  case 15:
#line 243 "function_parser.y" /* yacc.c:1646  */
    {
    		// printf("non-arithmetic regular logarithm: log_{%s} (%s)\n", $4->str().c_str(), $6->str().c_str());
   		/* vector <operand <double>> vals;
		
		vals.push_back(*$4);
		vals.push_back(*$6);

		$$ = new operand <double> (defaults <double> ::log_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back((fvsp[-2].expr));
		leaves.push_back((fvsp[0].expr));

		(fval.felm) = new functor <double> ::node {&defaults <double> ::log_op,
			functor <double> ::m_none, leaves};
}
#line 1533 "function_parser.h" /* yacc.c:1646  */
    break;

  case 16:
#line 259 "function_parser.y" /* yacc.c:1646  */
    { // Binary log
    		//printf("non-arithmetic binary logarithm of %s\n", $2->str().c_str());
   		/* vector <operand <double>> vals;
		
		vals.push_back(operand <double> (2));
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::log_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back(new functor <double> ::node (new operand <double> (2), functor <double> ::m_none,
			vector <functor <double> ::node *> ()));
		leaves.push_back((fvsp[0].expr));

		(fval.felm) = new functor <double> ::node {&defaults <double> ::log_op,
			functor <double> ::m_none, leaves};
}
#line 1554 "function_parser.h" /* yacc.c:1646  */
    break;

  case 17:
#line 276 "function_parser.y" /* yacc.c:1646  */
    { // Natural log
    		//printf("non-arithmetic natural logarithm of %s\n", $2->tok->str().c_str());
   		/* vector <operand <double>> vals;
		
		vals.push_back(operand <double> (exp(1.0)));
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::log_op(vals));
		vector <functor <double> ::node *> leaves;
		leaves.push_back($2);

		$$ = new functor <double> ::node {&defaults <double> ::sin_op,
			m_none, leaves}; */
		vector <functor <double> ::node *> leaves;
		leaves.push_back(new functor <double> ::node (new operand <double> (exp(1)), functor <double> ::m_none,
			vector <functor <double> ::node *> ()));
		leaves.push_back((fvsp[0].expr));

		(fval.felm) = new functor <double> ::node {&defaults <double> ::log_op,
			functor <double> ::m_none, leaves};
}
#line 1580 "function_parser.h" /* yacc.c:1646  */
    break;

  case 18:
#line 298 "function_parser.y" /* yacc.c:1646  */
    { // Log base 10
   		/* vector <operand <double>> vals;
		
		vals.push_back(operand <double> (10));
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::log_op(vals));
		vector <functor <double> ::node *> leaves;
		leaves.push_back($2);

		$$ = new functor <double> ::node {&defaults <double> ::sin_op,
			functor <double> ::m_none, leaves}; */
		vector <functor <double> ::node *> leaves;
		leaves.push_back(new functor <double> ::node (new operand <double> (10), functor <double> ::m_none,
			vector <functor <double> ::node *> ()));
		leaves.push_back((fvsp[0].expr));

		(fval.felm) = new functor <double> ::node {&defaults <double> ::log_op,
			functor <double> ::m_none, leaves};
}
#line 1605 "function_parser.h" /* yacc.c:1646  */
    break;

  case 19:
#line 319 "function_parser.y" /* yacc.c:1646  */
    { // Cot
   		/* vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::cot_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back((fvsp[0].expr));

		(fval.felm) = new functor <double> ::node {&defaults <double> ::cot_op,
			functor <double> ::m_none, leaves};
}
#line 1621 "function_parser.h" /* yacc.c:1646  */
    break;

  case 20:
#line 331 "function_parser.y" /* yacc.c:1646  */
    { // Sec
   		/* vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::sec_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back((fvsp[0].expr));

		(fval.felm) = new functor <double> ::node {&defaults <double> ::sec_op,
			functor <double> ::m_none, leaves};
}
#line 1637 "function_parser.h" /* yacc.c:1646  */
    break;

  case 21:
#line 343 "function_parser.y" /* yacc.c:1646  */
    { // Csc
   		/* vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::csc_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back((fvsp[0].expr));

		(fval.felm) = new functor <double> ::node {&defaults <double> ::csc_op,
			functor <double> ::m_none, leaves};
}
#line 1653 "function_parser.h" /* yacc.c:1646  */
    break;

  case 22:
#line 355 "function_parser.y" /* yacc.c:1646  */
    { // Tan
   		/* vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::tan_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back((fvsp[0].expr));

		(fval.felm) = new functor <double> ::node {&defaults <double> ::tan_op,
			functor <double> ::m_none, leaves};
}
#line 1669 "function_parser.h" /* yacc.c:1646  */
    break;

  case 23:
#line 367 "function_parser.y" /* yacc.c:1646  */
    { // Cos
   		/* vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::cos_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back((fvsp[0].expr));

		(fval.felm) = new functor <double> ::node {&defaults <double> ::cos_op,
			functor <double> ::m_none, leaves};
}
#line 1685 "function_parser.h" /* yacc.c:1646  */
    break;

  case 24:
#line 379 "function_parser.y" /* yacc.c:1646  */
    { // Sin
		/* vector <operand <double>> vals;
		vals.push_back(*$2);

		$$ = new operand <double> (defaults <double> ::sin_op(vals)); */
		vector <functor <double> ::node *> leaves;
		leaves.push_back((fvsp[0].expr));

		(fval.felm) = new functor <double> ::node {&defaults <double> ::sin_op,
			functor <double> ::m_none, leaves};
}
#line 1701 "function_parser.h" /* yacc.c:1646  */
    break;

  case 25:
#line 392 "function_parser.y" /* yacc.c:1646  */
    {
		/* vector <operand <double>> vals;
		vals.push_back(*$1);
		vals.push_back(*$3);

		$$ = new operand <double> (defaults <double> ::exp_op(vals)); */
		
		vector <functor <double> ::node *> leaves;
		leaves.push_back((fvsp[-2].dopn));
		leaves.push_back((fvsp[0].dopn));

		(fval.dopn) = new functor <double> ::node {&defaults <double> ::exp_op,
			functor <double> ::m_none, leaves};
}
#line 1720 "function_parser.h" /* yacc.c:1646  */
    break;

  case 26:
#line 407 "function_parser.y" /* yacc.c:1646  */
    {
    		(fval.dopn) = (fvsp[0].dpnt);
}
#line 1728 "function_parser.h" /* yacc.c:1646  */
    break;

  case 27:
#line 411 "function_parser.y" /* yacc.c:1646  */
    {
    		(fval.dopn) = (fvsp[0].sclr);
}
#line 1736 "function_parser.h" /* yacc.c:1646  */
    break;

  case 28:
#line 415 "function_parser.y" /* yacc.c:1646  */
    {
    		(fval.dopn) = (fvsp[0].prth);
}
#line 1744 "function_parser.h" /* yacc.c:1646  */
    break;

  case 29:
#line 420 "function_parser.y" /* yacc.c:1646  */
    { // Variable
    		//printf("dependant, variable %s\n", $1);
		string str = (fvsp[0].ident);
		
		/* try {
			var = vst.find(str);
		} catch (...) {
			ferror(value, vst, "no variable in scope");
		} */

		/* bool param = false;
		for (auto v : list) {
			if (v.symbol() == str) {
				param = true;
				break;
			}
		}

		if (!param)
			ferror(root, list, vmap, "no variable in function scope"); */

		// variable <double> *var = new variable <double> {str, true};
		functor <double> ::node *save;
		functor <double> ::node *temp;
		functor <double> ::node *in;
		// variable <double> var;

		(fval.dpnt) = new functor <double> ::node {&defaults <double> ::mult_op,
			functor <double> ::m_variable, vector <functor <double> ::node *>
			{new functor <double> ::node {new operand <double> (1), functor <double> ::m_none, {}},
			new functor <double> ::node {new operand <double> (1), functor <double> ::m_none, {}}}};
		temp = (fval.dpnt);

		int num = 0;

		string acc;
		for (int i = 0; i < str.length(); i++) {
			acc += str[i];

			auto var = find_if(list.begin(), list.end(),
				[&](const variable <double> &vr) {
					return vr.symbol() == acc;
				}
			);

			if (var != list.end()) {
				/* in = new functor <double> ::node {&(*var), functor <double> ::m_none, {}};
				temp->leaves[1] = new functor <double> ::node {&defaults <double> ::mult_op,
					functor <double> ::m_none, vector <functor <double> ::node *> {in,
					new functor <double> ::node {new operand <double> (1), functor <double> ::m_none, {}}}};
				temp = temp->leaves[1]; */
				(fval.dpnt) = new functor <double> ::node {&defaults <double> ::mult_op, functor <double> ::m_none,
					{(fval.dpnt), new functor <double> ::node {new variable <double> {var->symbol(), true}, functor <double> ::m_none, {}}}};
				vmap[var->symbol()].push_back((fval.dpnt)->leaves[1]);
				// temp = $$;
				acc.clear();
				num++;
			}
		}

		// printf("done\n");
		//$$ = new functor <double> ::node {new variable <double> {"x", true}, functor <double> ::m_none, {}};
		//functor <double> ::print($$, 1, 0);

		if (!num)
			throw functor <double> ::invalid_definition();

		// operand <double> val = new operand <double> (var.get());
}
#line 1818 "function_parser.h" /* yacc.c:1646  */
    break;

  case 30:
#line 491 "function_parser.y" /* yacc.c:1646  */
    { // Number
		operand <double> *val = new operand <double> ((fvsp[0].value));
		(fval.sclr) = new functor <double> ::node {val, functor <double> ::m_constant,
			vector <functor <double> ::node *> ()};
    		//printf("scalar, %s\n", $$->tok->str().c_str());
}
#line 1829 "function_parser.h" /* yacc.c:1646  */
    break;

  case 31:
#line 499 "function_parser.y" /* yacc.c:1646  */
    { // Parenthesis
    		//printf("parenthesis, %s\n", $2->tok->str().c_str());
   		(fval.prth) = (fvsp[-1].expr);
}
#line 1838 "function_parser.h" /* yacc.c:1646  */
    break;


#line 1842 "function_parser.h" /* yacc.c:1646  */
      default: break;
    }
  /* User semantic actions sometimes alter fchar, and that requires
     that ftoken be updated with the new translation.  We take the
     approach of translating immediately before every use of ftoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     FABORT, FACCEPT, or FERROR immediately after altering fchar or
     if it invokes FBACKUP.  In the case of FABORT or FACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of FERROR or FBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  F_SYMBOL_PRINT ("-> $$ =", fr1[fn], &fval, &floc);

  FPOPSTACK (flen);
  flen = 0;
  F_STACK_PRINT (fss, fssp);

  *++fvsp = fval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  fn = fr1[fn];

  fstate = fpgoto[fn - FNTOKENS] + *fssp;
  if (0 <= fstate && fstate <= FLAST && fcheck[fstate] == *fssp)
    fstate = ftable[fstate];
  else
    fstate = fdefgoto[fn - FNTOKENS];

  goto fnewstate;


/*--------------------------------------.
| ferrlab -- here on detecting error.  |
`--------------------------------------*/
ferrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  ftoken = fchar == FEMPTY ? FEMPTY : FTRANSLATE (fchar);

  /* If not already recovering from an error, report this error.  */
  if (!ferrstatus)
    {
      ++fnerrs;
#if ! FERROR_VERBOSE
      ferror (root, list, vmap, F_("syntax error"));
#else
# define FSYNTAX_ERROR fsyntax_error (&fmsg_alloc, &fmsg, \
                                        fssp, ftoken)
      {
        char const *fmsgp = F_("syntax error");
        int fsyntax_error_status;
        fsyntax_error_status = FSYNTAX_ERROR;
        if (fsyntax_error_status == 0)
          fmsgp = fmsg;
        else if (fsyntax_error_status == 1)
          {
            if (fmsg != fmsgbuf)
              FSTACK_FREE (fmsg);
            fmsg = (char *) FSTACK_ALLOC (fmsg_alloc);
            if (!fmsg)
              {
                fmsg = fmsgbuf;
                fmsg_alloc = sizeof fmsgbuf;
                fsyntax_error_status = 2;
              }
            else
              {
                fsyntax_error_status = FSYNTAX_ERROR;
                fmsgp = fmsg;
              }
          }
        ferror (root, list, vmap, fmsgp);
        if (fsyntax_error_status == 2)
          goto fexhaustedlab;
      }
# undef FSYNTAX_ERROR
#endif
    }



  if (ferrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (fchar <= FEOF)
        {
          /* Return failure if at end of input.  */
          if (fchar == FEOF)
            FABORT;
        }
      else
        {
          fdestruct ("Error: discarding",
                      ftoken, &flval, root, list, vmap);
          fchar = FEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto ferrlab1;


/*---------------------------------------------------.
| ferrorlab -- error raised explicitly by FERROR.  |
`---------------------------------------------------*/
ferrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     FERROR and the label ferrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto ferrorlab;

  /* Do not reclaim the symbols of the rule whose action triggered
     this FERROR.  */
  FPOPSTACK (flen);
  flen = 0;
  F_STACK_PRINT (fss, fssp);
  fstate = *fssp;
  goto ferrlab1;


/*-------------------------------------------------------------.
| ferrlab1 -- common code for both syntax error and FERROR.  |
`-------------------------------------------------------------*/
ferrlab1:
  ferrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      fn = fpact[fstate];
      if (!fpact_value_is_default (fn))
        {
          fn += FTERROR;
          if (0 <= fn && fn <= FLAST && fcheck[fn] == FTERROR)
            {
              fn = ftable[fn];
              if (0 < fn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (fssp == fss)
        FABORT;


      fdestruct ("Error: popping",
                  fstos[fstate], fvsp, root, list, vmap);
      FPOPSTACK (1);
      fstate = *fssp;
      F_STACK_PRINT (fss, fssp);
    }

  F_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++fvsp = flval;
  F_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  F_SYMBOL_PRINT ("Shifting", fstos[fn], fvsp, flsp);

  fstate = fn;
  goto fnewstate;


/*-------------------------------------.
| facceptlab -- FACCEPT comes here.  |
`-------------------------------------*/
facceptlab:
  fresult = 0;
  goto freturn;

/*-----------------------------------.
| fabortlab -- FABORT comes here.  |
`-----------------------------------*/
fabortlab:
  fresult = 1;
  goto freturn;

#if !defined foverflow || FERROR_VERBOSE
/*-------------------------------------------------.
| fexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
fexhaustedlab:
  ferror (root, list, vmap, F_("memory exhausted"));
  fresult = 2;
  /* Fall through.  */
#endif

freturn:
  if (fchar != FEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      ftoken = FTRANSLATE (fchar);
      fdestruct ("Cleanup: discarding lookahead",
                  ftoken, &flval, root, list, vmap);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this FABORT or FACCEPT.  */
  FPOPSTACK (flen);
  F_STACK_PRINT (fss, fssp);
  while (fssp != fss)
    {
      fdestruct ("Cleanup: popping",
                  fstos[*fssp], fvsp, root, list, vmap);
      FPOPSTACK (1);
    }
#ifndef foverflow
  if (fss != fssa)
    FSTACK_FREE (fss);
#endif
#if FERROR_VERBOSE
  if (fmsg != fmsgbuf)
    FSTACK_FREE (fmsg);
#endif
  return fresult;
}
#line 504 "function_parser.y" /* yacc.c:1906  */


void ferror(functor <double> ::node *(&nd), functor <double> ::param_list l, functor <double> ::map m, const char *error)
{
	cout << error << endl;
}
