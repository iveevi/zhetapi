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

/* All symbols defined below should begin with e or E, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define EBISON 1

/* Bison version.  */
#define EBISON_VERSION "3.0.4"

/* Skeleton name.  */
#define ESKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define EPURE 0

/* Push parsers.  */
#define EPUSH 0

/* Pull parsers.  */
#define EPULL 1

/* Substitute the type names.
#define ESTYPE         ESTYPE
* Substitute the variable and function names.
#define eparse         eparse
#define elex           elex
#define eerror         eerror
#define edebug         edebug
#define enerrs         enerrs

#define elval          elval
#define echar          echar */

/* Copy the first part of user declarations.  */
#line 1 "expression_parser.y" /* yacc.c:339  */

	#include <iostream>
	#include <cctype>
	#include <cstring>
	#include <vector>
	#include <stack>
	#include <string>

	#include "operation.h"
	#include "defaults.h"
  	#include "operand.h"
	
	#include "var_stack.h"
	#include "variable.h"

	extern "C" int elex();
  	
	using namespace std;

  	template <class T>
	void eerror (operand <T> *, var_stack <T>, const char *error);

#line 98 "expression_parser.h" /* yacc.c:339  */

# ifndef E_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define E_NULLPTR nullptr
#  else
#   define E_NULLPTR 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef EERROR_VERBOSE
# undef EERROR_VERBOSE
# define EERROR_VERBOSE 1
#else
# define EERROR_VERBOSE 1
#endif


/* Debug traces.  */
#ifndef EDEBUG
# if defined EDEBUG
#if EDEBUG
#   define EDEBUG 1
#  else
#   define EDEBUG 0
#  endif
# else /* ! defined EDEBUG */
#  define EDEBUG 0
# endif /* ! defined EDEBUG */
#endif  /* ! defined EDEBUG */
#if EDEBUG
extern int edebug;
#endif

/* Token type.  */
#ifndef ETOKENTYPE
# define ETOKENTYPE
  enum etokentype
  {
    E_IDENT = 258,
    E_NUMBER = 259,
    E_PLUS = 260,
    E_MINUS = 261,
    E_MULT = 262,
    E_DIV = 263,
    E_SIN = 264,
    E_COS = 265,
    E_TAN = 266,
    E_CSC = 267,
    E_SEC = 268,
    E_COT = 269,
    E_LOG = 270,
    E_LN = 271,
    E_LG = 272,
    E_SUPERSCRIPT = 273,
    E_SUBSCRIPT = 274,
    E_LPAREN = 275,
    E_RPAREN = 276,
    E_LBRACE = 277,
    E_RBRACE = 278,
    E_LBRACKET = 279,
    E_RBRACKET = 280,
    E_END = 281,
    DIV = 282,
    TAN = 283
  };
#endif

/* Value type.  */
#if ! defined ESTYPE && ! defined ESTYPE_IS_DECLARED

union ESTYPE
{
#line 52 "expression_parser.y" /* yacc.c:355  */

	operand <double>		*expr;
	operand <double>		*coll;
	operand <double>		*term;
	operand <double>		*felm;
	operand <double>		*dopn;
	operand <double>		*dpnt;
	operand <double>		*prth;
	operand <double>		*sclr;

	const char			*ident;
	double       			value;

#line 186 "expression_parser.h" /* yacc.c:355  */
};

typedef union ESTYPE ESTYPE;
# define ESTYPE_IS_TRIVIAL 1
# define ESTYPE_IS_DECLARED 1
#endif


extern ESTYPE elval;

int eparse (operand <double> *value, var_stack <double> vst);



/* Copy the second part of user declarations.  */

#line 203 "expression_parser.h" /* yacc.c:358  */

#ifdef short
# undef short
#endif

#ifdef ETYPE_UINT8
typedef ETYPE_UINT8 etype_uint8;
#else
typedef unsigned char etype_uint8;
#endif

#ifdef ETYPE_INT8
typedef ETYPE_INT8 etype_int8;
#else
typedef signed char etype_int8;
#endif

#ifdef ETYPE_UINT16
typedef ETYPE_UINT16 etype_uint16;
#else
typedef unsigned short int etype_uint16;
#endif

#ifdef ETYPE_INT16
typedef ETYPE_INT16 etype_int16;
#else
typedef short int etype_int16;
#endif

#ifndef ESIZE_T
# ifdef __SIZE_TYPE__
#  define ESIZE_T __SIZE_TYPE__
# elif defined size_t
#  define ESIZE_T size_t
# elif ! defined ESIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define ESIZE_T size_t
# else
#  define ESIZE_T unsigned int
# endif
#endif

#define ESIZE_MAXIMUM ((ESIZE_T) -1)

#ifndef E_
# if defined EENABLE_NLS && EENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define E_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef E_
#  define E_(Msgid) Msgid
# endif
#endif

#ifndef E_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define E_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define E_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef E_ATTRIBUTE_PURE
# define E_ATTRIBUTE_PURE   E_ATTRIBUTE ((__pure__))
#endif

#ifndef E_ATTRIBUTE_UNUSED
# define E_ATTRIBUTE_UNUSED E_ATTRIBUTE ((__unused__))
#endif

#if !defined _Noreturn \
     && (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112)
# if defined _MSC_VER && 1200 <= _MSC_VER
#  define _Noreturn __declspec (noreturn)
# else
#  define _Noreturn E_ATTRIBUTE ((__noreturn__))
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define EUSE(E) ((void) (E))
#else
# define EUSE(E) /* empty */
#endif

#if defined __GNUC__ && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about elval being uninitialized.  */
# define E_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define E_IGNORE_MAYBE_UNINITIALIZED_END \
    _Pragma ("GCC diagnostic pop")
#else
# define E_INITIAL_VALUE(Value) Value
#endif
#ifndef E_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define E_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define E_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef E_INITIAL_VALUE
# define E_INITIAL_VALUE(Value) /* Nothing. */
#endif


#if ! defined eoverflow || EERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef ESTACK_USE_ALLOCA
#  if ESTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define ESTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define ESTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define ESTACK_ALLOC alloca
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

# ifdef ESTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define ESTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef ESTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define ESTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define ESTACK_ALLOC EMALLOC
#  define ESTACK_FREE EFREE
#  ifndef ESTACK_ALLOC_MAXIMUM
#   define ESTACK_ALLOC_MAXIMUM ESIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined EMALLOC || defined malloc) \
             && (defined EFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef EMALLOC
#   define EMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (ESIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef EFREE
#   define EFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined eoverflow || EERROR_VERBOSE */


#if (! defined eoverflow \
     && (! defined __cplusplus \
         || (defined ESTYPE_IS_TRIVIAL && ESTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union ealloc
{
  etype_int16 ess_alloc;
  ESTYPE evs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define ESTACK_GAP_MAXIMUM (sizeof (union ealloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define ESTACK_BYTES(N) \
     ((N) * (sizeof (etype_int16) + sizeof (ESTYPE)) \
      + ESTACK_GAP_MAXIMUM)

# define ECOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables ESIZE and ESTACKSIZE give the old and new number of
   elements in the stack, and EPTR gives the new location of the
   stack.  Advance EPTR to a properly aligned location for the next
   stack.  */
# define ESTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        ESIZE_T enewbytes;                                            \
        ECOPY (&eptr->Stack_alloc, Stack, esize);                    \
        Stack = &eptr->Stack_alloc;                                    \
        enewbytes = estacksize * sizeof (*Stack) + ESTACK_GAP_MAXIMUM; \
        eptr += enewbytes / sizeof (*eptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined ECOPY_NEEDED && ECOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef ECOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define ECOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define ECOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          ESIZE_T ei;                         \
          for (ei = 0; ei < (Count); ei++)   \
            (Dst)[ei] = (Src)[ei];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !ECOPY_NEEDED */

/* EFINAL -- State number of the termination state.  */
#define EFINAL  35
/* ELAST -- Last index in ETABLE.  */
#define ELAST   114

/* ENTOKENS -- Number of terminals.  */
#define ENTOKENS  29
/* ENNTS -- Number of nonterminals.  */
#define ENNTS  10
/* ENRULES -- Number of rules.  */
#define ENRULES  31
/* ENSTATES -- Number of states.  */
#define ENSTATES  56

/* ETRANSLATE[EX] -- Symbol number corresponding to EX as returned
   by elex, with out-of-bounds checking.  */
#define EUNDEFTOK  2
#define EMAXUTOK   283

#define ETRANSLATE(EX)                                                \
  ((unsigned int) (EX) <= EMAXUTOK ? etranslate[EX] : EUNDEFTOK)

/* ETRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by elex, without out-of-bounds checking.  */
static const etype_uint8 etranslate[] =
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
      25,    26,    27,    28
};

#if EDEBUG
  /* ERLINE[EN] -- Source line where rule number EN was defined.  */
static const etype_uint16 erline[] =
{
       0,    98,    98,   104,   113,   122,   131,   140,   149,   158,
     164,   173,   177,   183,   191,   196,   206,   216,   226,   235,
     242,   249,   256,   263,   270,   278,   286,   290,   294,   299,
     314,   320
};
#endif

#if EDEBUG || EERROR_VERBOSE || 1
/* ETNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at ENTOKENS, nonterminals.  */
static const char *const etname[] =
{
  "$end", "error", "$undefined", "E_IDENT", "E_NUMBER", "E_PLUS",
  "E_MINUS", "E_MULT", "E_DIV", "E_SIN", "E_COS", "E_TAN", "E_CSC",
  "E_SEC", "E_COT", "E_LOG", "E_LN", "E_LG", "E_SUPERSCRIPT",
  "E_SUBSCRIPT", "E_LPAREN", "E_RPAREN", "E_LBRACE", "E_RBRACE",
  "E_LBRACKET", "E_RBRACKET", "E_END", "DIV", "TAN", "$accept", "input",
  "expr", "coll", "term", "felm", "dopn", "dpnt", "sclr", "prth", E_NULLPTR
};
#endif

# ifdef EPRINT
/* ETOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const etype_uint16 etoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283
};
# endif

#define EPACT_NINF -25

#define epact_value_is_default(Yystate) \
  (!!((Yystate) == (-25)))

#define ETABLE_NINF -1

#define etable_value_is_error(Yytable_value) \
  0

  /* EPACT[STATE-NUM] -- Index in ETABLE of the portion describing
     STATE-NUM.  */
static const etype_int8 epact[] =
{
      79,   -25,   -25,    94,    79,    79,    79,    79,    79,    79,
      61,    79,    79,    79,    10,    12,   -25,    94,   -25,    -5,
     -25,   -25,   -25,   -25,   -25,   -25,   -25,   -25,   -25,   -25,
     -10,   -25,   -25,   -25,    22,   -25,    79,    79,    79,    79,
     -25,    79,    49,   -25,    49,    79,   -25,    24,    24,    -3,
      -3,    -3,    -5,    39,    79,   -25
};

  /* EDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when ETABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const etype_uint8 edefact[] =
{
       0,    29,    30,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     9,    12,    11,    14,
      26,    27,    28,     8,    24,    23,    22,    21,    20,    19,
       0,    18,    17,    16,     0,     1,     0,     0,     0,     0,
       2,     0,    13,    10,     0,     0,    31,     6,     7,     4,
       3,     5,    25,     0,     0,    15
};

  /* EPGOTO[NTERM-NUM].  */
static const etype_int8 epgoto[] =
{
     -25,   -25,    -4,    11,    -6,    -1,   -24,   -25,   -25,   -25
};

  /* EDEFGOTO[NTERM-NUM].  */
static const etype_int8 edefgoto[] =
{
      -1,    14,    15,    16,    17,    18,    19,    20,    21,    22
};

  /* ETABLE[EPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If ETABLE_NINF, syntax error.  */
static const etype_uint8 etable[] =
{
      24,    25,    26,    27,    28,    29,    31,    32,    33,    34,
      35,    42,    45,    44,    23,    39,    43,    36,    37,    38,
      52,     0,     0,     0,     0,     0,     0,    36,    37,    38,
      39,    38,    47,    48,    49,    50,    42,    51,    40,    41,
      39,    53,    39,    46,    36,    37,    38,     0,     0,    41,
      55,    41,     1,     2,     0,     0,     0,    39,     0,     0,
       0,     0,    54,     0,     1,     2,    41,     3,     0,    13,
       4,     5,     6,     7,     8,     9,    10,    11,    12,     0,
      30,    13,     1,     2,     0,     3,     0,     0,     4,     5,
       6,     7,     8,     9,    10,    11,    12,     1,     2,    13,
       0,     0,     0,     4,     5,     6,     7,     8,     9,    10,
      11,    12,     0,     0,    13
};

static const etype_int8 echeck[] =
{
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
       0,    17,    22,    18,     3,    18,    17,     5,     6,     7,
      44,    -1,    -1,    -1,    -1,    -1,    -1,     5,     6,     7,
      18,     7,    36,    37,    38,    39,    42,    41,    26,    27,
      18,    45,    18,    21,     5,     6,     7,    -1,    -1,    27,
      54,    27,     3,     4,    -1,    -1,    -1,    18,    -1,    -1,
      -1,    -1,    23,    -1,     3,     4,    27,     6,    -1,    20,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    -1,
      19,    20,     3,     4,    -1,     6,    -1,    -1,     9,    10,
      11,    12,    13,    14,    15,    16,    17,     3,     4,    20,
      -1,    -1,    -1,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    -1,    -1,    20
};

  /* ESTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const etype_uint8 estos[] =
{
       0,     3,     4,     6,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    20,    30,    31,    32,    33,    34,    35,
      36,    37,    38,    32,    31,    31,    31,    31,    31,    31,
      19,    31,    31,    31,    31,     0,     5,     6,     7,    18,
      26,    27,    33,    34,    18,    22,    21,    31,    31,    31,
      31,    31,    35,    31,    23,    31
};

  /* ER1[EN] -- Symbol number of symbol that rule EN derives.  */
static const etype_uint8 er1[] =
{
       0,    29,    30,    31,    31,    31,    31,    31,    31,    31,
      32,    32,    32,    33,    33,    34,    34,    34,    34,    34,
      34,    34,    34,    34,    34,    35,    35,    35,    35,    36,
      37,    38
};

  /* ER2[EN] -- Number of symbols on the right hand side of rule EN.  */
static const etype_uint8 er2[] =
{
       0,     2,     2,     3,     3,     3,     3,     3,     2,     1,
       2,     1,     1,     2,     1,     6,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     3,     1,     1,     1,     1,
       1,     3
};


#define eerrok         (eerrstatus = 0)
#define eclearin       (echar = EEMPTY)
#define EEMPTY         (-2)
#define EEOF           0

#define EACCEPT        goto eacceptlab
#define EABORT         goto eabortlab
#define EERROR         goto eerrorlab


#define ERECOVERING()  (!!eerrstatus)

#define EBACKUP(Token, Value)                                  \
do                                                              \
  if (echar == EEMPTY)                                        \
    {                                                           \
      echar = (Token);                                         \
      elval = (Value);                                         \
      EPOPSTACK (elen);                                       \
      estate = *essp;                                         \
      goto ebackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      eerror (value, vst, E_("syntax error: cannot back up")); \
      EERROR;                                                  \
    }                                                           \
while (0)

/* Error token number */
#define ETERROR        1
#define EERRCODE       256



/* Enable debugging if requested.  */
#if EDEBUG

# ifndef EFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define EFPRINTF fprintf
# endif

# define EDPRINTF(Args)                        \
do {                                            \
  if (edebug)                                  \
    EFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
#ifndef E_LOCATION_PRINT
# define E_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


# define E_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (edebug)                                                            \
    {                                                                     \
      EFPRINTF (stderr, "%s ", Title);                                   \
      e_symbol_print (stderr,                                            \
                  Type, Value, value, vst); \
      EFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*----------------------------------------.
| Print this symbol's value on EOUTPUT.  |
`----------------------------------------*/

static void
e_symbol_value_print (FILE *eoutput, int etype, ESTYPE const * const evaluep, operand <double> *value, var_stack <double> vst)
{
  FILE *eo = eoutput;
  EUSE (eo);
  EUSE (value);
  EUSE (vst);
  if (!evaluep)
    return;
# ifdef EPRINT
  if (etype < ENTOKENS)
    EPRINT (eoutput, etoknum[etype], *evaluep);
# endif
  EUSE (etype);
}


/*--------------------------------.
| Print this symbol on EOUTPUT.  |
`--------------------------------*/

static void
e_symbol_print (FILE *eoutput, int etype, ESTYPE const * const evaluep, operand <double> *value, var_stack <double> vst)
{
  EFPRINTF (eoutput, "%s %s (",
             etype < ENTOKENS ? "token" : "nterm", etname[etype]);

  e_symbol_value_print (eoutput, etype, evaluep, value, vst);
  EFPRINTF (eoutput, ")");
}

/*------------------------------------------------------------------.
| e_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
e_stack_print (etype_int16 *ebottom, etype_int16 *etop)
{
  EFPRINTF (stderr, "Stack now");
  for (; ebottom <= etop; ebottom++)
    {
      int ebot = *ebottom;
      EFPRINTF (stderr, " %d", ebot);
    }
  EFPRINTF (stderr, "\n");
}

# define E_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (edebug)                                                  \
    e_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the ERULE is going to be reduced.  |
`------------------------------------------------*/

static void
e_reduce_print (etype_int16 *essp, ESTYPE *evsp, int erule, operand <double> *value, var_stack <double> vst)
{
  unsigned long int elno = erline[erule];
  int enrhs = er2[erule];
  int ei;
  EFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             erule - 1, elno);
  /* The symbols being reduced.  */
  for (ei = 0; ei < enrhs; ei++)
    {
      EFPRINTF (stderr, "   $%d = ", ei + 1);
      e_symbol_print (stderr,
                       estos[essp[ei + 1 - enrhs]],
                       &(evsp[(ei + 1) - (enrhs)])
                                              , value, vst);
      EFPRINTF (stderr, "\n");
    }
}

# define E_REDUCE_PRINT(Rule)          \
do {                                    \
  if (edebug)                          \
    e_reduce_print (essp, evsp, Rule, value, vst); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int edebug;
#else /* !EDEBUG */
# define EDPRINTF(Args)
# define E_SYMBOL_PRINT(Title, Type, Value, Location)
# define E_STACK_PRINT(Bottom, Top)
# define E_REDUCE_PRINT(Rule)
#endif /* !EDEBUG */


/* EINITDEPTH -- initial size of the parser's stacks.  */
#ifndef EINITDEPTH
# define EINITDEPTH 200
#endif

/* EMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   ESTACK_ALLOC_MAXIMUM < ESTACK_BYTES (EMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef EMAXDEPTH
# define EMAXDEPTH 10000
#endif


#if EERROR_VERBOSE

# ifndef estrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define estrlen strlen
#  else
/* Return the length of ESTR.  */
static ESIZE_T
estrlen (const char *estr)
{
  ESIZE_T elen;
  for (elen = 0; estr[elen]; elen++)
    continue;
  return elen;
}
#  endif
# endif

# ifndef estpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define estpcpy stpcpy
#  else
/* Copy ESRC to EDEST, returning the address of the terminating '\0' in
   EDEST.  */
static char *
estpcpy (char *edest, const char *esrc)
{
  char *ed = edest;
  const char *es = esrc;

  while ((*ed++ = *es++) != '\0')
    continue;

  return ed - 1;
}
#  endif
# endif

# ifndef etnamerr
/* Copy to ERES the contents of ESTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for eerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  ESTR is taken from etname.  If ERES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static ESIZE_T
etnamerr (char *eres, const char *estr)
{
  if (*estr == '"')
    {
      ESIZE_T en = 0;
      char const *ep = estr;

      for (;;)
        switch (*++ep)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++ep != '\\')
              goto do_not_strip_quotes;
            /* Fall through.  */
          default:
            if (eres)
              eres[en] = *ep;
            en++;
            break;

          case '"':
            if (eres)
              eres[en] = '\0';
            return en;
          }
    do_not_strip_quotes: ;
    }

  if (! eres)
    return estrlen (estr);

  return estpcpy (eres, estr) - eres;
}
# endif

/* Copy into *EMSG, which is of size *EMSG_ALLOC, an error message
   about the unexpected token ETOKEN for the state stack whose top is
   ESSP.

   Return 0 if *EMSG was successfully written.  Return 1 if *EMSG is
   not large enough to hold the message.  In that case, also set
   *EMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
esyntax_error (ESIZE_T *emsg_alloc, char **emsg,
                etype_int16 *essp, int etoken)
{
  ESIZE_T esize0 = etnamerr (E_NULLPTR, etname[etoken]);
  ESIZE_T esize = esize0;
  enum { EERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *eformat = E_NULLPTR;
  /* Arguments of eformat. */
  char const *earg[EERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int ecount = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in echar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated echar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (etoken != EEMPTY)
    {
      int en = epact[*essp];
      earg[ecount++] = etname[etoken];
      if (!epact_value_is_default (en))
        {
          /* Start EX at -EN if negative to avoid negative indexes in
             ECHECK.  In other words, skip the first -EN actions for
             this state because they are default actions.  */
          int exbegin = en < 0 ? -en : 0;
          /* Stay within bounds of both echeck and etname.  */
          int echecklim = ELAST - en + 1;
          int exend = echecklim < ENTOKENS ? echecklim : ENTOKENS;
          int ex;

          for (ex = exbegin; ex < exend; ++ex)
            if (echeck[ex + en] == ex && ex != ETERROR
                && !etable_value_is_error (etable[ex + en]))
              {
                if (ecount == EERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    ecount = 1;
                    esize = esize0;
                    break;
                  }
                earg[ecount++] = etname[ex];
                {
                  ESIZE_T esize1 = esize + etnamerr (E_NULLPTR, etname[ex]);
                  if (! (esize <= esize1
                         && esize1 <= ESTACK_ALLOC_MAXIMUM))
                    return 2;
                  esize = esize1;
                }
              }
        }
    }

  switch (ecount)
    {
# define ECASE_(N, S)                      \
      case N:                               \
        eformat = S;                       \
      break
      ECASE_(0, E_("syntax error"));
      ECASE_(1, E_("syntax error, unexpected %s"));
      ECASE_(2, E_("syntax error, unexpected %s, expecting %s"));
      ECASE_(3, E_("syntax error, unexpected %s, expecting %s or %s"));
      ECASE_(4, E_("syntax error, unexpected %s, expecting %s or %s or %s"));
      ECASE_(5, E_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef ECASE_
    }

  {
    ESIZE_T esize1 = esize + estrlen (eformat);
    if (! (esize <= esize1 && esize1 <= ESTACK_ALLOC_MAXIMUM))
      return 2;
    esize = esize1;
  }

  if (*emsg_alloc < esize)
    {
      *emsg_alloc = 2 * esize;
      if (! (esize <= *emsg_alloc
             && *emsg_alloc <= ESTACK_ALLOC_MAXIMUM))
        *emsg_alloc = ESTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *ep = *emsg;
    int ei = 0;
    while ((*ep = *eformat) != '\0')
      if (*ep == '%' && eformat[1] == 's' && ei < ecount)
        {
          ep += etnamerr (ep, earg[ei++]);
          eformat += 2;
        }
      else
        {
          ep++;
          eformat++;
        }
  }
  return 0;
}
#endif /* EERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
edestruct (const char *emsg, int etype, ESTYPE *evaluep, operand <double> *value, var_stack <double> vst)
{
  EUSE (evaluep);
  EUSE (value);
  EUSE (vst);
  if (!emsg)
    emsg = "Deleting";
  E_SYMBOL_PRINT (emsg, etype, evaluep, elocationp);

  E_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  EUSE (etype);
  E_IGNORE_MAYBE_UNINITIALIZED_END
}




/* The lookahead symbol.  */
int echar;

/* The semantic value of the lookahead symbol.  */
ESTYPE elval;
/* Number of syntax errors so far.  */
int enerrs;


/*----------.
| eparse.  |
`----------*/

int
eparse (operand <double> *value, var_stack <double> vst)
{
    int estate;
    /* Number of tokens to shift before error messages enabled.  */
    int eerrstatus;

    /* The stacks and their tools:
       'ess': related to states.
       'evs': related to semantic values.

       Refer to the stacks through separate pointers, to allow eoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    etype_int16 essa[EINITDEPTH];
    etype_int16 *ess;
    etype_int16 *essp;

    /* The semantic value stack.  */
    ESTYPE evsa[EINITDEPTH];
    ESTYPE *evs;
    ESTYPE *evsp;

    ESIZE_T estacksize;

  int en;
  int eresult;
  /* Lookahead token as an internal (translated) token number.  */
  int etoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  ESTYPE eval;

#if EERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char emsgbuf[128];
  char *emsg = emsgbuf;
  ESIZE_T emsg_alloc = sizeof emsgbuf;
#endif

#define EPOPSTACK(N)   (evsp -= (N), essp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int elen = 0;

  essp = ess = essa;
  evsp = evs = evsa;
  estacksize = EINITDEPTH;

  EDPRINTF ((stderr, "Starting parse\n"));

  estate = 0;
  eerrstatus = 0;
  enerrs = 0;
  echar = EEMPTY; /* Cause a token to be read.  */
  goto esetstate;

/*------------------------------------------------------------.
| enewstate -- Push a new state, which is found in estate.  |
`------------------------------------------------------------*/
 enewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  essp++;

 esetstate:
  *essp = estate;

  if (ess + estacksize - 1 <= essp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      ESIZE_T esize = essp - ess + 1;

#ifdef eoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        ESTYPE *evs1 = evs;
        etype_int16 *ess1 = ess;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if eoverflow is a macro.  */
        eoverflow (E_("memory exhausted"),
                    &ess1, esize * sizeof (*essp),
                    &evs1, esize * sizeof (*evsp),
                    &estacksize);

        ess = ess1;
        evs = evs1;
      }
#else /* no eoverflow */
# ifndef ESTACK_RELOCATE
      goto eexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (EMAXDEPTH <= estacksize)
        goto eexhaustedlab;
      estacksize *= 2;
      if (EMAXDEPTH < estacksize)
        estacksize = EMAXDEPTH;

      {
        etype_int16 *ess1 = ess;
        union ealloc *eptr =
          (union ealloc *) ESTACK_ALLOC (ESTACK_BYTES (estacksize));
        if (! eptr)
          goto eexhaustedlab;
        ESTACK_RELOCATE (ess_alloc, ess);
        ESTACK_RELOCATE (evs_alloc, evs);
#  undef ESTACK_RELOCATE
        if (ess1 != essa)
          ESTACK_FREE (ess1);
      }
# endif
#endif /* no eoverflow */

      essp = ess + esize - 1;
      evsp = evs + esize - 1;

      EDPRINTF ((stderr, "Stack size increased to %lu\n",
                  (unsigned long int) estacksize));

      if (ess + estacksize - 1 <= essp)
        EABORT;
    }

  EDPRINTF ((stderr, "Entering state %d\n", estate));

  if (estate == EFINAL)
    EACCEPT;

  goto ebackup;

/*-----------.
| ebackup.  |
`-----------*/
ebackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  en = epact[estate];
  if (epact_value_is_default (en))
    goto edefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* ECHAR is either EEMPTY or EEOF or a valid lookahead symbol.  */
  if (echar == EEMPTY)
    {
      EDPRINTF ((stderr, "Reading a token: "));
      echar = elex ();
    }

  if (echar <= EEOF)
    {
      echar = etoken = EEOF;
      EDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      etoken = ETRANSLATE (echar);
      E_SYMBOL_PRINT ("Next token is", etoken, &elval, &elloc);
    }

  /* If the proper action on seeing token ETOKEN is to reduce or to
     detect an error, take that action.  */
  en += etoken;
  if (en < 0 || ELAST < en || echeck[en] != etoken)
    goto edefault;
  en = etable[en];
  if (en <= 0)
    {
      if (etable_value_is_error (en))
        goto eerrlab;
      en = -en;
      goto ereduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (eerrstatus)
    eerrstatus--;

  /* Shift the lookahead token.  */
  E_SYMBOL_PRINT ("Shifting", etoken, &elval, &elloc);

  /* Discard the shifted token.  */
  echar = EEMPTY;

  estate = en;
  E_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++evsp = elval;
  E_IGNORE_MAYBE_UNINITIALIZED_END

  goto enewstate;


/*-----------------------------------------------------------.
| edefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
edefault:
  en = edefact[estate];
  if (en == 0)
    goto eerrlab;
  goto ereduce;


/*-----------------------------.
| ereduce -- Do a reduction.  |
`-----------------------------*/
ereduce:
  /* en is the number of a rule to reduce with.  */
  elen = er2[en];

  /* If ELEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets EVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to EVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that EVAL may be used uninitialized.  */
  eval = evsp[1-elen];


  E_REDUCE_PRINT (en);
  switch (en)
    {
        case 2:
#line 98 "expression_parser.y" /* yacc.c:1646  */
    {
     		value->set((evsp[-1].expr)->get());
		return 0;
}
#line 1330 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 3:
#line 104 "expression_parser.y" /* yacc.c:1646  */
    { // Exponentiation
   		printf("expression exponentiation\n");
		vector <operand <double>> vals;
		vals.push_back(*(evsp[-2].expr));
		vals.push_back(*(evsp[0].expr));

		(eval.expr) = new operand <double> (defaults <double> ::exp_op(vals));
}
#line 1343 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 4:
#line 113 "expression_parser.y" /* yacc.c:1646  */
    { // Multiplication
   		printf("expression multiplication\n");
		vector <operand <double>> vals;
		vals.push_back(*(evsp[-2].expr));
		vals.push_back(*(evsp[0].expr));

		(eval.expr) = new operand <double> (defaults <double> ::mult_op(vals));
}
#line 1356 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 5:
#line 122 "expression_parser.y" /* yacc.c:1646  */
    { // Division
   		printf("expression divition\n");
		vector <operand <double>> vals;
		vals.push_back(*(evsp[-2].expr));
		vals.push_back(*(evsp[0].expr));

		(eval.expr) = new operand <double> (defaults <double> ::div_op(vals));
}
#line 1369 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 6:
#line 131 "expression_parser.y" /* yacc.c:1646  */
    { // Addition
   		printf("expression addition\n");
		vector <operand <double>> vals;
		vals.push_back(*(evsp[-2].expr));
		vals.push_back(*(evsp[0].expr));

		(eval.expr) = new operand <double> (defaults <double> ::add_op(vals));
}
#line 1382 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 7:
#line 140 "expression_parser.y" /* yacc.c:1646  */
    { // Subtraction
   		printf("expression substraction\n");
   		vector <operand <double>> vals;
		vals.push_back(*(evsp[-2].expr));
		vals.push_back(*(evsp[0].expr));

		(eval.expr) = new operand <double> (defaults <double> ::sub_op(vals));
}
#line 1395 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 8:
#line 149 "expression_parser.y" /* yacc.c:1646  */
    {
   		printf("expression negative collective\n");
   		vector <operand <double>> vals;
		vals.push_back(operand <double> (-1));
		vals.push_back(*(evsp[0].coll));

		(eval.expr) = new operand <double> (defaults <double> ::mult_op(vals));
}
#line 1408 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 9:
#line 158 "expression_parser.y" /* yacc.c:1646  */
    {
   		printf("expression collective\n");
   		(eval.expr) = (evsp[0].coll);
}
#line 1417 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 10:
#line 164 "expression_parser.y" /* yacc.c:1646  */
    { // Implicit Multiplication: term and non-arithmetic operation
    		printf("collective, term (%s) implicitly multiplicied with non-arithmetic operation (%s)\n", (evsp[-1].term)->str().c_str(), (evsp[0].felm)->str().c_str());
		vector <operand <double>> vals;
		vals.push_back(*(evsp[-1].term));
		vals.push_back(*(evsp[0].felm));

		(eval.coll) = new operand <double> (defaults <double> ::mult_op(vals));
}
#line 1430 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 11:
#line 173 "expression_parser.y" /* yacc.c:1646  */
    {
    		(eval.coll) = (evsp[0].felm);
}
#line 1438 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 12:
#line 177 "expression_parser.y" /* yacc.c:1646  */
    {
    		printf("collective as a regular term (%s)\n", (evsp[0].term)->str().c_str());
    		(eval.coll) = (evsp[0].term);
}
#line 1447 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 13:
#line 183 "expression_parser.y" /* yacc.c:1646  */
    { // Implicit Multiplication: two or more terms
		vector <operand <double>> vals;
		vals.push_back(*(evsp[-1].term));
		vals.push_back(*(evsp[0].term));

		(eval.term) = new operand <double> (defaults <double> ::mult_op(vals));
}
#line 1459 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 14:
#line 191 "expression_parser.y" /* yacc.c:1646  */
    { // Direct Operand
    		(eval.term) = (evsp[0].dopn);
}
#line 1467 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 15:
#line 196 "expression_parser.y" /* yacc.c:1646  */
    {
    		printf("non-arithmetic regular logarithm: log_{%s} (%s)\n", (evsp[-2].expr)->str().c_str(), (evsp[0].expr)->str().c_str());
   		vector <operand <double>> vals;
		
		vals.push_back(*(evsp[-2].expr));
		vals.push_back(*(evsp[0].expr));

		(eval.felm) = new operand <double> (defaults <double> ::log_op(vals));
}
#line 1481 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 16:
#line 206 "expression_parser.y" /* yacc.c:1646  */
    { // Binary log
    		printf("non-arithmetic binary logarithm of %s\n", (evsp[0].expr)->str().c_str());
   		vector <operand <double>> vals;
		
		vals.push_back(operand <double> (2));
		vals.push_back(*(evsp[0].expr));

		(eval.felm) = new operand <double> (defaults <double> ::log_op(vals));
}
#line 1495 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 17:
#line 216 "expression_parser.y" /* yacc.c:1646  */
    { // Natural log
    		printf("non-arithmetic natural logarithm of %s\n", (evsp[0].expr)->str().c_str());
   		vector <operand <double>> vals;
		
		vals.push_back(operand <double> (exp(1.0)));
		vals.push_back(*(evsp[0].expr));

		(eval.felm) = new operand <double> (defaults <double> ::log_op(vals));
}
#line 1509 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 18:
#line 226 "expression_parser.y" /* yacc.c:1646  */
    { // Log base 10
   		vector <operand <double>> vals;
		
		vals.push_back(operand <double> (10));
		vals.push_back(*(evsp[0].expr));

		(eval.felm) = new operand <double> (defaults <double> ::log_op(vals));
}
#line 1522 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 19:
#line 235 "expression_parser.y" /* yacc.c:1646  */
    { // Cot
   		vector <operand <double>> vals;
		vals.push_back(*(evsp[0].expr));

		(eval.felm) = new operand <double> (defaults <double> ::cot_op(vals));
}
#line 1533 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 20:
#line 242 "expression_parser.y" /* yacc.c:1646  */
    { // Sec
   		vector <operand <double>> vals;
		vals.push_back(*(evsp[0].expr));

		(eval.felm) = new operand <double> (defaults <double> ::sec_op(vals));
}
#line 1544 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 21:
#line 249 "expression_parser.y" /* yacc.c:1646  */
    { // Csc
   		vector <operand <double>> vals;
		vals.push_back(*(evsp[0].expr));

		(eval.felm) = new operand <double> (defaults <double> ::csc_op(vals));
}
#line 1555 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 22:
#line 256 "expression_parser.y" /* yacc.c:1646  */
    { // Tan
   		vector <operand <double>> vals;
		vals.push_back(*(evsp[0].expr));

		(eval.felm) = new operand <double> (defaults <double> ::tan_op(vals));
}
#line 1566 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 23:
#line 263 "expression_parser.y" /* yacc.c:1646  */
    { // Cos
   		vector <operand <double>> vals;
		vals.push_back(*(evsp[0].expr));

		(eval.felm) = new operand <double> (defaults <double> ::cos_op(vals));
}
#line 1577 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 24:
#line 270 "expression_parser.y" /* yacc.c:1646  */
    { // Sin
		vector <operand <double>> vals;
		vals.push_back(*(evsp[0].expr));

		(eval.felm) = new operand <double> (defaults <double> ::sin_op(vals));
}
#line 1588 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 25:
#line 278 "expression_parser.y" /* yacc.c:1646  */
    {
		vector <operand <double>> vals;
		vals.push_back(*(evsp[-2].dopn));
		vals.push_back(*(evsp[0].dopn));

		(eval.dopn) = new operand <double> (defaults <double> ::exp_op(vals));
}
#line 1600 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 26:
#line 286 "expression_parser.y" /* yacc.c:1646  */
    {
    		(eval.dopn) = (evsp[0].dpnt);
}
#line 1608 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 27:
#line 290 "expression_parser.y" /* yacc.c:1646  */
    {
    		(eval.dopn) = (evsp[0].sclr);
}
#line 1616 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 28:
#line 294 "expression_parser.y" /* yacc.c:1646  */
    {
    		(eval.dopn) = (evsp[0].prth);
}
#line 1624 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 29:
#line 299 "expression_parser.y" /* yacc.c:1646  */
    { // Variable
    		printf("dependant, variable %s\n", (evsp[0].ident));
		variable <double> var;
		string str = (evsp[0].ident);
		
		try {
			var = vst.find(str);
		} catch (...) {
			eerror(value, vst, "no variable in scope");
		}

		(eval.dpnt) = new operand <double> (var.get());
}
#line 1642 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 30:
#line 314 "expression_parser.y" /* yacc.c:1646  */
    { // Number
		(eval.sclr) = new operand <double> ((evsp[0].value));
    		printf("scalar, %s\n", (eval.sclr)->str().c_str());
}
#line 1651 "expression_parser.h" /* yacc.c:1646  */
    break;

  case 31:
#line 320 "expression_parser.y" /* yacc.c:1646  */
    { // Parenthesis
    		printf("parenthesis, %s\n", (evsp[-1].expr)->str().c_str());
   		(eval.prth) = (evsp[-1].expr);
}
#line 1660 "expression_parser.h" /* yacc.c:1646  */
    break;


#line 1664 "expression_parser.h" /* yacc.c:1646  */
      default: break;
    }
  /* User semantic actions sometimes alter echar, and that requires
     that etoken be updated with the new translation.  We take the
     approach of translating immediately before every use of etoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     EABORT, EACCEPT, or EERROR immediately after altering echar or
     if it invokes EBACKUP.  In the case of EABORT or EACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of EERROR or EBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  E_SYMBOL_PRINT ("-> $$ =", er1[en], &eval, &eloc);

  EPOPSTACK (elen);
  elen = 0;
  E_STACK_PRINT (ess, essp);

  *++evsp = eval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  en = er1[en];

  estate = epgoto[en - ENTOKENS] + *essp;
  if (0 <= estate && estate <= ELAST && echeck[estate] == *essp)
    estate = etable[estate];
  else
    estate = edefgoto[en - ENTOKENS];

  goto enewstate;


/*--------------------------------------.
| eerrlab -- here on detecting error.  |
`--------------------------------------*/
eerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  etoken = echar == EEMPTY ? EEMPTY : ETRANSLATE (echar);

  /* If not already recovering from an error, report this error.  */
  if (!eerrstatus)
    {
      ++enerrs;
#if ! EERROR_VERBOSE
      eerror (value, vst, E_("syntax error"));
#else
# define ESYNTAX_ERROR esyntax_error (&emsg_alloc, &emsg, \
                                        essp, etoken)
      {
        char const *emsgp = E_("syntax error");
        int esyntax_error_status;
        esyntax_error_status = ESYNTAX_ERROR;
        if (esyntax_error_status == 0)
          emsgp = emsg;
        else if (esyntax_error_status == 1)
          {
            if (emsg != emsgbuf)
              ESTACK_FREE (emsg);
            emsg = (char *) ESTACK_ALLOC (emsg_alloc);
            if (!emsg)
              {
                emsg = emsgbuf;
                emsg_alloc = sizeof emsgbuf;
                esyntax_error_status = 2;
              }
            else
              {
                esyntax_error_status = ESYNTAX_ERROR;
                emsgp = emsg;
              }
          }
        eerror (value, vst, emsgp);
        if (esyntax_error_status == 2)
          goto eexhaustedlab;
      }
# undef ESYNTAX_ERROR
#endif
    }



  if (eerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (echar <= EEOF)
        {
          /* Return failure if at end of input.  */
          if (echar == EEOF)
            EABORT;
        }
      else
        {
          edestruct ("Error: discarding",
                      etoken, &elval, value, vst);
          echar = EEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto eerrlab1;


/*---------------------------------------------------.
| eerrorlab -- error raised explicitly by EERROR.  |
`---------------------------------------------------*/
eerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     EERROR and the label eerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto eerrorlab;

  /* Do not reclaim the symbols of the rule whose action triggered
     this EERROR.  */
  EPOPSTACK (elen);
  elen = 0;
  E_STACK_PRINT (ess, essp);
  estate = *essp;
  goto eerrlab1;


/*-------------------------------------------------------------.
| eerrlab1 -- common code for both syntax error and EERROR.  |
`-------------------------------------------------------------*/
eerrlab1:
  eerrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      en = epact[estate];
      if (!epact_value_is_default (en))
        {
          en += ETERROR;
          if (0 <= en && en <= ELAST && echeck[en] == ETERROR)
            {
              en = etable[en];
              if (0 < en)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (essp == ess)
        EABORT;


      edestruct ("Error: popping",
                  estos[estate], evsp, value, vst);
      EPOPSTACK (1);
      estate = *essp;
      E_STACK_PRINT (ess, essp);
    }

  E_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++evsp = elval;
  E_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  E_SYMBOL_PRINT ("Shifting", estos[en], evsp, elsp);

  estate = en;
  goto enewstate;


/*-------------------------------------.
| eacceptlab -- EACCEPT comes here.  |
`-------------------------------------*/
eacceptlab:
  eresult = 0;
  goto ereturn;

/*-----------------------------------.
| eabortlab -- EABORT comes here.  |
`-----------------------------------*/
eabortlab:
  eresult = 1;
  goto ereturn;

#if !defined eoverflow || EERROR_VERBOSE
/*-------------------------------------------------.
| eexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
eexhaustedlab:
  eerror (value, vst, E_("memory exhausted"));
  eresult = 2;
  /* Fall through.  */
#endif

ereturn:
  if (echar != EEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      etoken = ETRANSLATE (echar);
      edestruct ("Cleanup: discarding lookahead",
                  etoken, &elval, value, vst);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this EABORT or EACCEPT.  */
  EPOPSTACK (elen);
  E_STACK_PRINT (ess, essp);
  while (essp != ess)
    {
      edestruct ("Cleanup: popping",
                  estos[*essp], evsp, value, vst);
      EPOPSTACK (1);
    }
#ifndef eoverflow
  if (ess != essa)
    ESTACK_FREE (ess);
#endif
#if EERROR_VERBOSE
  if (emsg != emsgbuf)
    ESTACK_FREE (emsg);
#endif
  return eresult;
}
#line 325 "expression_parser.y" /* yacc.c:1906  */


template <class T>
void eerror (operand <T> *optr, var_stack <T> vst, const char *error)
{
	cout << error << endl;
}
