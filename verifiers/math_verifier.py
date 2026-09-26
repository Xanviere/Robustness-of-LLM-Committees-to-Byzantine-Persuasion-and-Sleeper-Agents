import re
import sympy
from typing import Optional

def normalize_latex(text: str) -> str:
    if not text:
        return ""
    s = str(text).strip()
    
    # Strip boxed and math delimiters
    s = re.sub(r'\\boxed\{([^}]+)\}', r'\1', s)
    s = s.replace('$', '').replace('\\(', '').replace('\\)', '').replace('\\[', '').replace('\\]', '')
    
    # Remove text wrappers
    s = re.sub(r'\\text\{([^}]+)\}', r'\1', s)
    s = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', s)
    s = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', s)
    
    # Convert square roots \sqrt{x} -> sqrt(x)
    s = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', s)
    s = re.sub(r'\\sqrt\[([^\]]+)\]\{([^}]+)\}', r'((\2)**(1/(\1)))', s)
    
    # LaTeX symbol normalization
    s = s.replace('\\left', '').replace('\\right', '')
    s = s.replace('\\cdot', '*').replace('\\times', '*').replace('^\\circ', '')
    s = s.replace('\\le', '<=').replace('\\ge', '>=')
    s = s.replace('\\leq', '<=').replace('\\geq', '>=')
    s = s.replace('\\neq', '!=')
    s = s.replace('\\infty', 'oo')
    s = s.replace('\\pi', 'pi')
    s = s.replace('^\\prime', "'")
    
    # Common functions
    for func in ['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan', 'log', 'ln', 'exp']:
        s = s.replace(f'\\{func}', func)

    # Convert \frac{a}{b} to ((a)/(b))
    for _ in range(5):
        s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', s)
        
    # Power conversion: ^{b} -> **(b)
    s = re.sub(r'\^\{([^}]+)\}', r'**(\1)', s)
    s = re.sub(r'\^([0-9a-zA-Z])', r'**\1', s)

    # Implicit multiplication e.g., 2pi -> 2*pi, 2(x) -> 2*(x)
    s = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', s)
    s = re.sub(r'(\))([0-9a-zA-Z(])', r'\1*\2', s)

    s = re.sub(r'\s+', '', s)
    return s

def verify_math(candidate: str, ground_truth: str) -> bool:
    norm_cand = normalize_latex(candidate)
    norm_gt = normalize_latex(ground_truth)

    if norm_cand == norm_gt:
        return True

    # SymPy symbolic equivalence check
    try:
        try:
            from sympy.parsing.latex import parse_latex
            c_expr = parse_latex(candidate.strip('$ '))
            g_expr = parse_latex(ground_truth.strip('$ '))
            if sympy.simplify(c_expr - g_expr) == 0:
                return True
        except Exception:
            pass

        c_sym = sympy.sympify(norm_cand)
        g_sym = sympy.sympify(norm_gt)
        diff = sympy.simplify(c_sym - g_sym)
        if diff == 0:
            return True
            
        if hasattr(diff, 'is_number') and diff.is_number:
            if abs(complex(diff.evalf())) < 1e-6:
                return True
    except Exception:
        pass

    return norm_cand.lower() == norm_gt.lower()
