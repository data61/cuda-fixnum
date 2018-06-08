#include "sexp.h"

#include <cstdlib>  // strtol is better than std::stol

using namespace std;

struct parser {
    explicit parser(istream &is_) : is(is_), pos(1) { };
    const sexp *parse_sexp();

private:
    static constexpr int OPEN_PAREN = (int)'(';
    static constexpr int CLOSE_PAREN = (int)')';
    static constexpr int LEN_DELIMIT = (int)':';
    static constexpr int MAX_LEN_DIGITS = 16;
    static constexpr int LEN_BASE = 10;

    istream &is;
    long pos;

    const sexp *parse_atom();
    const sexp *parse_list();

    void die_if(bool p, const char *msg) {
        if (p) {
            cerr << "Parse Error near byte " << pos << ": " << msg << endl;
            abort();
        }
    }
};

const sexp *parser::parse_atom() {
    char buf[MAX_LEN_DIGITS], *e;
    long bytesread, atomlen, bufbytes;
    atom *a;

    is.getline(buf, MAX_LEN_DIGITS, LEN_DELIMIT);
    die_if( ! is, "atom length delimiter not found");
    bytesread = (long) is.gcount();

    atomlen = strtol(buf, &e, LEN_BASE);
    // +1 because istream::gcount() counts the delimiter
    bufbytes = e - buf + 1;
    pos += bufbytes;
    die_if(bufbytes != bytesread, "non-numeric characters in length field");
    die_if(atomlen < 0, "negative length field");

    a = new atom;
    a->data.resize(atomlen);
    is.read(reinterpret_cast<char *>(a->data.data()), atomlen);
    pos += is.gcount();
    die_if( ! is, "error extracting byte array");

    return a;
}

const sexp *parser::parse_list() {
    list *lst;
    int c;

    c = is.get();
    die_if( ! is || c != OPEN_PAREN, "expected open parenthesis");
    ++pos;
    lst = new list;
    while (is && is.get() != CLOSE_PAREN) {
        const sexp *s;

        is.unget();
        s = parse_sexp();
        lst->sexps.push_back(s);
    }
    die_if( ! is, "expected close parenthesis");
    ++pos;
    return lst;
}

const sexp *parser::parse_sexp() {
    int c = is.peek();
    die_if(c == EOF, "unexpected EOF");
    return (c == OPEN_PAREN) ? parse_list() : parse_atom();
}

const sexp *sexp::create(istream &is) {
    return parser(is).parse_sexp();
}
