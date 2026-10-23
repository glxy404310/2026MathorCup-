"""Small, deliberately bounded TeX-to-OMML path for this manuscript."""
import re
from lxml import etree

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def el(tag, *children, **attrs):
    node = etree.Element(f"{{{M}}}{tag}")
    for key, value in attrs.items():
        node.set(f"{{{M}}}{key}", str(value))
    node.extend(children)
    return node


def run(text, plain=False):
    node = el("r")
    if plain:
        node.append(el("rPr", el("sty", val="p")))
    pr = etree.SubElement(node, f"{{{W}}}rPr")
    fonts = etree.SubElement(pr, f"{{{W}}}rFonts")
    for key in ["ascii", "hAnsi", "eastAsia"]:
        fonts.set(f"{{{W}}}{key}", "STIX Two Math")
    size = etree.SubElement(pr, f"{{{W}}}sz")
    size.set(f"{{{W}}}val", "22")
    t = etree.SubElement(node, f"{{{M}}}t")
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return node


SYMBOLS = dict(alpha="α", beta="β", gamma="γ", delta="δ", epsilon="ε", varepsilon="ε",
               lambda_="λ", mu="μ", pi="π", rho="ρ", sigma="σ", tau="τ", kappa="κ",
               Phi="Φ", Sigma="Σ", Delta="Δ", Omega="Ω", ell="ℓ", in_="∈",
               notin="∉", neq="≠", ne="≠", leq="≤", le="≤", geq="≥", ge="≥",
               times="×", cdot="·", cap="∩", cup="∪", forall="∀", prime="′",
               ldots="…", dots="…", lVert="‖", rVert="‖", vert="|", Vert="‖",
               to="→", rightarrow="→", approx="≈", sim="∼", infinity="∞", infty="∞",
               sum="∑", prod="∏", partial="∂", pm="±", mid="|", mathbfone="1")
SYMBOLS["lambda"] = SYMBOLS.pop("lambda_")
SYMBOLS["in"] = SYMBOLS.pop("in_")


class Parser:
    def __init__(self, source):
        source = re.sub(r"\\{2,}", r"\\", source)
        self.tokens = re.findall(r"\\[A-Za-z]+|\\.|.", source)
        self.i = 0
        self.unknown = []

    def take(self):
        value = self.tokens[self.i]
        self.i += 1
        return value

    def argument(self):
        while self.i < len(self.tokens) and self.tokens[self.i].isspace():
            self.i += 1
        if self.i >= len(self.tokens):
            return []
        if self.tokens[self.i] == "{":
            self.i += 1
            return self.sequence("}")
        return self.atom()

    def atom(self):
        tok = self.take()
        if tok == "{":
            return self.sequence("}")
        if not tok.startswith("\\"):
            return [run(tok)]
        cmd = tok[1:]
        if cmd in ["left", "right", "big", "Big", "bigg", "Bigg"]:
            return []
        if cmd in [",", ";", "quad", "qquad", " ", "!"]:
            return [run(" " if cmd != "!" else "")]
        if cmd in ["{", "}", "_", "%", "|", "[", "]"]:
            return [run(cmd)]
        if cmd in ["frac", "dfrac", "tfrac"]:
            return [el("f", el("num", *self.argument()), el("den", *self.argument()))]
        if cmd == "sqrt":
            return [el("rad", el("radPr", el("degHide", val="1")), el("deg"), el("e", *self.argument()))]
        if cmd in ["hat", "widehat", "tilde", "widetilde", "bar", "overline"]:
            chars = {"hat": "̂", "widehat": "̂", "tilde": "̃", "widetilde": "̃", "bar": "̄", "overline": "̄"}
            return [el("acc", el("accPr", el("chr", val=chars[cmd])), el("e", *self.argument()))]
        if cmd in ["mathrm", "operatorname", "text", "mathsf", "mathbf", "boldsymbol", "mathcal", "mathbb"]:
            nodes = self.argument()
            for node in nodes:
                for r in [node] if node.tag == f"{{{M}}}r" else node.iter(f"{{{M}}}r"):
                    pr = r.find(f"{{{M}}}rPr")
                    if pr is None:
                        pr = el("rPr")
                        r.insert(0, pr)
                    pr.append(el("sty", val="b" if cmd in ["mathbf", "boldsymbol"] else "p"))
                    if cmd == "mathcal":
                        pr.append(el("scr", val="script"))
            return [el("box", el("e", *nodes))]
        if cmd in ["max", "min", "arg", "log", "exp", "sin", "cos", "rank", "corr", "argmin"]:
            return [run(cmd, plain=True)]
        if cmd in SYMBOLS:
            return [run(SYMBOLS[cmd])]
        self.unknown.append(cmd)
        return [run(cmd, plain=True)]

    def sequence(self, until=None):
        nodes = []
        while self.i < len(self.tokens):
            tok = self.tokens[self.i]
            if tok == until:
                self.i += 1
                break
            if tok in ["_", "^"]:
                self.i += 1
                sub = self.argument() if tok == "_" else None
                sup = self.argument() if tok == "^" else None
                if self.i < len(self.tokens) and self.tokens[self.i] in ["_", "^"]:
                    other = self.take()
                    if other == "_":
                        sub = self.argument()
                    else:
                        sup = self.argument()
                base = nodes.pop() if nodes else run("")
                text = "".join(base.itertext())
                if text in ["∑", "∏"]:
                    props = el("naryPr", el("chr", val=text), el("limLoc", val="undOvr"))
                    if sub is None:
                        props.append(el("subHide", val="1"))
                    if sup is None:
                        props.append(el("supHide", val="1"))
                    nodes.append(el("nary", props, el("sub", *(sub or [])), el("sup", *(sup or [])), el("e")))
                elif sub is not None and sup is not None:
                    nodes.append(el("sSubSup", el("e", base), el("sub", *sub), el("sup", *sup)))
                elif sub is not None:
                    nodes.append(el("sSub", el("e", base), el("sub", *sub)))
                else:
                    nodes.append(el("sSup", el("e", base), el("sup", *(sup or []))))
            else:
                nodes.extend(self.atom())
        return nodes


def to_omml(source):
    parser = Parser(source)
    result = el("oMath", *parser.sequence())
    # An OMML n-ary operator must own a non-empty expression; otherwise Word
    # displays an editing placeholder square. Attach its following atom.
    for parent in list(result.iter())[::-1]:
        for child in list(parent)[::-1]:
            if child.tag != f"{{{M}}}nary":
                continue
            expression=child.find(f"{{{M}}}e")
            if len(expression):
                continue
            following=child.getnext()
            while following is not None and not "".join(following.itertext()).strip():
                next_node=following.getnext()
                parent.remove(following)
                following=next_node
            if following is not None:
                parent.remove(following)
                expression.append(following)
            else:
                expression.append(run("1"))
    if parser.unknown:
        raise ValueError(f"Unsupported mathematical command(s): {parser.unknown}")
    return result

