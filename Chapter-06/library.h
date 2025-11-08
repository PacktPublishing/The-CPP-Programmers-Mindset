#ifndef C6_LIBRARY_H
#define C6_LIBRARY_H


#include "mysharedlibrary_export.h"

namespace ct {
class MYSHAREDLIBRARY_EXPORT MyExportedClass {
public:
    void exported_method();
};


MYSHAREDLIBRARY_EXPORT int exported_function(int arg);

namespace dtl {
class UnexportedInternalClass;
}
}


#endif //C6_LIBRARY_H