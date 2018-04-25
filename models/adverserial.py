

def case1():
    p = "SSA  will  consider  the  comments  received  by  April  14  ,  1997  ,  and  will  issue  revised  regulations  if  necessary.    "
    h = "The  regulations  will  be  revised  as  necessary  after  the  SSA  considers  the  comments  received  before  April  14  ,  1997  ."
    h_keyword = "revised"
    h_key_replace_syn = ["revised","altered", "amended", "improved", "modified", "reorganized", "updated"] # 0 ~ 6
    h_key_replace_ant = ["harmed", "ignored", "ruined"] # 6~8
    h_key_replace_conf= ["discarded", "rejected", "abandoned", "canceled", "eliminated", "removed"] # 9 ~
    return p,h, h_keyword, h_key_replace_syn, h_key_replace_ant, h_key_replace_conf

def case2():
    p = "Today  the  strait  is  busy  with  commercial  shipping  ,  ferries  ,  and  fishing  boats  ,  and  its  wooded  shores  are  lined  with  pretty  fishing  villages  ,  old  Ottoman  mansions  ,  and  the  villas  of  Istanbul  's  wealthier  citizens  . "
    h = "Today  ,  the  strait  is  empty  ."
    h_keyword = "empty"
    h_key_replace_ent = ["crowded", "populous", "jammed"] # 0 ~ 5
    h_key_replace_neu = []
    h_key_replace_con = ["vacant", "deserted", "barren", "left" ] # 6~8
    return p,h, h_keyword, h_key_replace_ent, h_key_replace_neu, h_key_replace_con

def case3():
    # Gold : E
    p = "However  ,  co-requesters  can  not  approve  additional  co-requesters  or  restrict  the  timing  of  the  release  of  the  product  after  it  is  issued  . "
    h = "They  can  not  restrict  timing  of  the  release  of  the  product  ."
    h_keyword = "release"
    h_key_replace_ent = ["issue", "open", "deploy"]  # 0 ~ 5
    h_key_replace_neu = ["production", "increase"]
    h_key_replace_con = []  # 6~8
    return p, h, h_keyword, h_key_replace_ent, h_key_replace_neu, h_key_replace_con


def case4():

    p = "yeah  that  's  where  i  got  to  too  the  first  i  got  chills  up  and  down  when  i  heard  the  on  the  radio  and  the  first  time  they  started  doing  the  bombing"
    h = "When  I  heard  that  on  the  radio  I  got  chills  and  the  beginning  of  the  bombing  . "
    h_keyword = "chills"
    h_key_replace_ent = ["cold", "freezing"]  # 0 ~ 5
    h_key_replace_neu = []
    h_key_replace_con = ["warm", "temperate", "hot", ]  #     return p,h, h_keyword, h_key_replace_syn, h_key_replace_ant, h_key_replace_conf
    return p, h, h_keyword, h_key_replace_ent, h_key_replace_neu, h_key_replace_con


def reorder():
    p = "Homes  or  businesses  not  located  on  one  of  these  roads  must  place  a  mail  receptacle  along  the  route  traveled  ."
    h1 = "Businesses or  Homes  not  located  on  one  of  these  roads  must  place  a  mail  receptacle  along  the  route  traveled  ."
    h2 = "Businesses or  Homes  located  on  one  of  these  roads  must  not  place  a  mail  receptacle  along  the  route  traveled  ."
    h3 = "Roads not  located  on  one  of  these  homes  or  businesses  must  place  a  mail  receptacle  along  the  route  traveled  ."
    h_entail = [h1]
    h_neutral = [h2, h3]

    return p, h_entail, h_neutral

def reorder2():
    p = "The  Committee  intends  that  LSC  consult  with  appropriate  stakeholders  in  developing  this  proposal  ."
    h1 = "The  stakeholders  discourage  LSC  to  consult  with  any  Committee  . "
    h2 = "The  Committee  discourage  stakeholders  to  consult  with  LSC  . "
    h_entail = []
    h_neutral = [h1, h2]

    return p, h_entail, h_neutral

def reorder3():
    p = "The  providers  worked  with  the  newly  created  Legal  Assistance  to  the  Disadvantaged  Committee  of  the  Minnesota  State  Bar  Association  -LRB-  MSBA  -RRB-  to  create  the  Minnesota  Legal  Services  Coalition  State  Support  Center  and  the  position  of  Director  of  Volunteer  Legal  Services  ,  now  the  Access  to  Justice  Director  at  the  Minnesota  State  Bar  Association  . "
    h1 = "The  Director  of  Volunteer  Legal  Services  was  formerly  called  the  Access  to  Justice  Director  . "
    h2 = "The  Access  to  Justice  Director  was  formerly  called  the  Services  of  Legal  Volunteer  Director. "
    h_entail = []
    h_neutral = [h1, h2]

    return p, h_entail, h_neutral


def antonym():
    ENTAILMENT = 0
    NEUTRAL = 1
    CONTRADICTION = 2


    def gen_e(args):
        p, h, h_keyword, h_key_replace_syn, h_key_replace_ant, h_key_replace_conf = args
        p_tokens = p.split(" ")
        h_tokens = h.split(" ")

        def gen_replace(ori_list, keyword, replaced_word):
            idx = ori_list.index(keyword)
            new_list = list(ori_list)
            new_list[idx] = replaced_word
            return new_list

        test_case = []
        for newword in h_key_replace_syn:
            test_case.append((p_tokens, gen_replace(h_tokens, h_keyword, newword), ENTAILMENT))
        for newword in h_key_replace_ant:
            test_case.append((p_tokens, gen_replace(h_tokens, h_keyword, newword), CONTRADICTION))
        for newword in h_key_replace_conf:
            test_case.append((p_tokens, gen_replace(h_tokens, h_keyword, newword), CONTRADICTION))

        return test_case

    def tag_e(args):
        p, h, h_keyword, h_key_replace_syn, h_key_replace_ant, h_key_replace_conf = args
        tags = []
        for newword in h_key_replace_syn:
            tags.append("ENTAIL_SYN_NC")
        for newword in h_key_replace_ant:
            tags.append("ENTAIL_ANT_C")
        for newword in h_key_replace_conf:
            tags.append("ENTAIL_CONF_C")

        return tags

    def tag_reorder(args):
        p, h_entail, h_neutral = args
        tags = []
        for newword in h_entail:
            tags.append("REORDER_NC")
        for newword in h_neutral:
            tags.append("REORDER_NEUTRAL")
        return tags


    def gen(args):
        p, h, h_keyword, h_key_replace_ent, h_key_replace_neu, h_key_replace_con = args
        p_tokens = p.split(" ")
        h_tokens = h.split(" ")

        def gen_replace(ori_list, keyword, replaced_word):
            idx = ori_list.index(keyword)
            new_list = list(ori_list)
            new_list[idx] = replaced_word
            return new_list

        test_case = []
        for newword in h_key_replace_ent:
            test_case.append((p_tokens, gen_replace(h_tokens, h_keyword, newword), ENTAILMENT))
        for newword in h_key_replace_neu:
            test_case.append((p_tokens, gen_replace(h_tokens, h_keyword, newword), NEUTRAL))
        for newword in h_key_replace_con:
            test_case.append((p_tokens, gen_replace(h_tokens, h_keyword, newword), CONTRADICTION))

        return test_case

    def tag_gen(args):
        p, h, h_keyword, h_key_replace_ent, h_key_replace_neu, h_key_replace_con = args
        tags = []
        for newword in h_key_replace_ent:
            tags.append("ENTAIL_SYN")
        for newword in h_key_replace_neu:
            tags.append("NEUTRAL_NON")
        for newword in h_key_replace_con:
            tags.append("CONTRADICTION_CONF")
        return tags

    def gen_simple(args):
        p, h_entail, h_neutral = args
        p_tokens = p.split(" ")

        test_case = []
        for h in h_entail:
            h_tokens = h.split(" ")
            test_case.append((p_tokens, h_tokens, ENTAILMENT))

        for h in h_neutral:
            h_tokens = h.split(" ")
            test_case.append((p_tokens, h_tokens, NEUTRAL))
        return test_case

    input = gen_e(case1())\
            + gen(case2())\
            + gen(case3())\
            + gen(case4())\
            + gen_simple(reorder())\
            + gen_simple(reorder2())\
            + gen_simple(reorder3())
    tag = tag_e(case1())\
          + tag_gen(case2())\
          + tag_gen(case3())\
          + tag_gen(case4())\
          + tag_reorder(reorder()) \
          + tag_reorder(reorder2()) \
          + tag_reorder(reorder3())

    assert(len(input)==len(tag))
    return input, tag
