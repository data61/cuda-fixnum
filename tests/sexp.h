#ifndef SEXP
#define SEXP

/*
 * Usage:
 *
 *  int main() {
 *      istream &is = cin;
 *      while (is && is.get() != EOF) {
 *          is.unget();
 *
 *          const sexp *s = sexp::create(is);
 *          myvisitor v;
 *          s->accept(v);
 *          delete s;
 *      }
 *  }
 */

#include <iostream>
#include <vector>
#include <stdint.h>

typedef std::vector<uint8_t> byte_array;

struct visitor;

struct sexp {
    static const sexp *create(std::istream &);
    virtual ~sexp() { }
    virtual void accept(visitor &v) const = 0;
};

struct atom;
struct list;

struct visitor {
    virtual void visit(const atom &) = 0;
    virtual void visit(const list &) = 0;
};

struct atom : sexp {
    byte_array data;

    virtual ~atom() { }
    virtual void accept(visitor &v) const { v.visit(*this); }

private:
    atom() { }
    friend class parser;
};

struct list : sexp {
    std::vector< const sexp * > sexps;

    virtual ~list() { for (auto s : sexps) delete s; }
    virtual void accept(visitor &v) const {
        v.visit(*this);
        for (auto s : sexps)
            s->accept(v);
    }

private:
    list() { };
    friend class parser;
};

#endif
