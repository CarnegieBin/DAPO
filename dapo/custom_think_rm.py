"""
Answer checker API that uses sympy to simplify expressions and check for equality.

Call grade_answer(given_answer: str, ground_truth: str).
"""
import re
from pylatexenc import latex2text
import sympy
from sympy.parsing import sympy_parser
from typing import Optional
from math_verify import parse, verify

# logging.info("DeepscaleR Here!!!")

# Dan Hendrycks' code
def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer

def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string


    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string


    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string


    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution

def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct

def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False

def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None

def _get_deepscaler_rule_base_reward(model_answer, label):
    if model_answer is None:
        return 0
    # logging.info("extract answer pass!!!")
    if label == "":
        return 0
    # logging.info("label pass!!!")
    # logging.info(f"Model Answer: {model_answer}, Label: {label}")
    # Convert single answer to list for uniform processing
    if isinstance(label, (str, float, int)):
        ground_truths = [label]
    else:
        print(f"ERROR GROUND TRUTH: {label}")
        return 0
        
    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)
    
    if not processed_ground_truths:
        return 0
    
    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
        if is_correct:
            return 1
            
    return 0
    
def verify_think_rm(data_source, solution_str, ground_truth, extra_info=None) -> dict:
    """Compute the reward score for a solution.
    
    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        config: Configuration object containing reward model settings
        pause_tokens_index: Indices of pause tokens
        
    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """

    think_count = solution_str.count("<think>")

    # 如果 <think> 多于一个，报错
    if think_count > 1:
        pred = "ERROR: Multiple <think>"
        acc = 0

    # 如果只有一个，但不在开头，也算异常
    elif think_count == 1 and not solution_str.strip().startswith("<think>"):
        pred = "ERROR: <think> not at beginning"
        acc = 0

    elif solution_str.count("</think>") != 1:
        pred = f"ERROR: Num of </think> == {solution_str.count('</think>')}"
        acc = 0
    else:
        model_solution = solution_str.split("</think>")[-1]
        pred = extract_answer(model_solution)
        if isinstance(ground_truth, list):
            acc = 0
            for truth in ground_truth:
                acc = _get_deepscaler_rule_base_reward(pred, truth)
                if acc == 1:
                    break
        else:
            acc = _get_deepscaler_rule_base_reward(pred, ground_truth)
        
        if pred is None:
            pred = "ERROR: Answer Extraction Failed"
            # print(pred)

    return {
        "score": acc,
        "acc": acc,
        "pred": pred,
    }


if __name__ == "__main__":
    solution = r"""<think>
I need to convert the point (0,3) from rectangular coordinates to polar coordinates. Polar coordinates are given as (r, θ), where r is the distance from the origin and θ is the angle from the positive x-axis. Both must be positive, and θ should be between 0 and 2π.
First, recall the formulas: for a point (x,y), r is given by r = √(x² + y²), and θ is given by θ = tan⁻¹(y/x), but I have to be careful with the quadrant.
Here, x = 0 and y = 3. So, plugging into the formula for r: r = √(0² + 3²) = √(0 + 9) = √9 = 3. So r is 3.
Now for θ. Since x is 0, the formula θ = tan⁻¹(y/x) is problematic because division by zero is undefined. When x is zero, the point is on the y-axis.
If x is zero and y is positive, that means the point is on the positive y-axis. So the angle should be π/2 radians, or 90 degrees.
Let me confirm. From the origin, going straight up to (0,3), that's indeed 90 degrees from the positive x-axis. So θ should be π/2.
The range is 0 ≤ θ < 2π, and π/2 is within that range.
I could think in terms of the arctangent function. The arctangent of y/x when x=0 and y>0 is undefined, but conventionally, we know that when x=0 and y>0, θ is π/2.
Similarly, if x=0 and y<0, θ would be 3π/2 or -π/2, but since the range is usually [0, 2π), it would be 3π/2.
But here y is positive, so π/2.
I can use the formula θ = atan2(y,x), which takes into account the signs of x and y to determine the correct quadrant.
In this case, atan2(3,0) should return π/2.
Yes, that makes sense.
So r is 3, θ is π/2.
The problem asks to enter the answer in the form (r, θ), so (3, π/2).
I should make sure that θ is in radians, since it's not specified, but in polar coordinates, especially in this context, radians are standard.
The range is given as 0 ≤ θ < 2π, which is typical for radians.
Degrees might be used sometimes, but here it's specified with 2π, so radians.
Also, r is positive, which it is.
Is there any other way to represent this point? For example, if I add 2π to θ, but 2π is not less than 2π, it's equal, but the range is 0 ≤ θ < 2π, so θ must be less than 2π, so π/2 is fine.
Sometimes people use negative angles, but here it's specified to be non-negative.
Another thing: when r is positive, θ can be adjusted by adding 2π, but again, not necessary here.
Or if r were negative, but here r is positive.
In this case, since it's on the y-axis, is there a unique representation? I think so, because r is positive and θ is defined uniquely in [0, 2π).
Sometimes people might think of θ as -3π/2 or something, but that's equivalent to 2π - 3π/2 = 4π/2 - 3π/2 = π/2, same thing.
But in standard form, we use the principal value.
So I think (3, π/2) is correct.
Let me write it down.
Point is (0,3).
r = √(x² + y²) = √(0 + 9) = 3.
θ = angle such that cosθ = x/r = 0/3 = 0, sinθ = y/r = 3/3 = 1.
So cosθ = 0 and sinθ = 1, which is θ = π/2.
Perfect.
I could use the formula, but it's straightforward.
So the polar coordinates are (3, π/2).
Now, the answer should be in the box.
I think that's it.
The problem says "enter your answer in the form (r, θ)", so I should write it as such.
Also, make sure it's clear.
Sometimes they might expect θ in degrees, but no, the range is given with 2π, so radians.
And r is positive, which it is.
So I think we're good.
Just to be thorough, let's see what the point would be if I used a different θ.
Suppose θ = π/2 + 2π = 5π/2, but 5π/2 is greater than 2π, and not in the range.
Or θ = -π/2, but -π/2 is not in [0, 2π).
And r would be negative, but if r is negative, θ is adjusted, but here r is positive.
So no.
Another way: sometimes for points on the axes, but in this case, it's clear.
So I believe the answer is (3, π/2).
Now, to box it.
The instruction is to put the final answer in \boxed{}, so I should do that.
So, \boxed{(3, \frac{\pi}{2})}
Or should I write it without the fraction? No, π/2 is standard.
Sometimes they write it as a decimal, but no, this is exact.
So I think that's fine.
Let me check the format.
It says "enter your answer in the form (r, θ)", so probably with parentheses.
Yes.
So, final answer.
</think>
The point (0,3) in rectangular coordinates has x = 0 and y = 3.
The distance r from the origin is given by r = √(x² + y²) = √(0² + 3²) = √9 = 3.
The angle θ is determined using the inverse tangent function, but since x = 0 and y > 0, the point lies on the positive y-axis. Thus, θ = π/2 radians.
The range for θ is 0 ≤ θ < 2π, and π/2 is within this range.
The polar coordinates are (3, π/2).
\boxed{(3,\ \dfrac{\pi}{2})}"""
    print(verify_think_rm('', solution, '\\left( 3, \\frac{\\pi}{2} \\right)'))