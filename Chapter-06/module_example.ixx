export module module_example;

export namespace ct {
// defined elsewhere
void exported_fn(int a);
}

// Not exported
int non_exported_fn();

// defined elsewhere
export int another_exported_fn(int b);

// exported inline function
export inline int inline_fn(int c) { return c; }

// defined elsewhere
export struct MyStruct;

export template<typename T>
T template_fn(T e) { return e; }

export template<typename T>
class TemplateClass {
    T f;

public:
    T get_f() { return f; }
};