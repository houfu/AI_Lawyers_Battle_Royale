import csv
from typing import Optional, Union

from langchain import PromptTemplate
from pydantic import BaseModel

scenario_template = """This is a court in the Supreme Court of Singapore.

These are the rules regarding {rule_title}.
```
{rule_content} 
{rule_secondary} 
```
Background: {case_background}
Application: {application}
"""


class Scenario(BaseModel):
    rule_title: str
    rule_content: str
    rule_secondary: Optional[str]
    case_background: str
    application: str


def load_scenarios() -> list[Scenario]:
    with open("court_rules.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        return [Scenario(**row) for row in reader]


scenarios = load_scenarios()


def get_scenario(scenario: Union[int, str]) -> Scenario:
    if type(scenario) == int:
        return scenarios[scenario]
    else:
        for test_scenario in scenarios:
            if test_scenario.rule_title == scenario:
                return test_scenario
    raise Exception()


scenario_prompt = PromptTemplate.from_template(scenario_template)

counsel_template = """You are counsel for the plaintiff. Your objective is to win by convincing the court to rule 
in your favour and rebutting the user's arguments. 

This is a court in the Supreme Court of Singapore.

These are the rules regarding {rule_title}.
```
{rule_content} 
{rule_secondary} 
```
Background: {case_background}
Application: {application}

In my messages, responses from the court start with `CT:` and responses from defendant's counsel start with `DC:`.
Do not generate responses for the court or the defendant's counsel.
"""

court_template = """You are an assistant registrar in the Supreme Court of Singapore. 
Your objective is to achieve the following ideals in civil procedure: (1) fair access to justice, 
(2) expeditious proceedings, (3) cost-effective work proportionate to the nature and importance of the action, 
complexity of the claim, (4) efficient use of court resources and (5) fair and practical results suited to the needs
of the parties.

These are the rules regarding {rule_title}.
```
{rule_content} 
{rule_secondary} 
```
Background: {case_background}
Application: {application}

Your role is to (1) give the plaintiff and defendant the fair opportunity to respond to all the points raised, and (2)
based on the arguments from counsels, render a decision. 
To direct a party to respond, end your response with the request and state `[PC]` for plaintiff's counsel to respond,
and `[DC]` for defendant's counsel to respond.
When the Defendant and Plaintiff have responded to all arguments, end arguments by rendering a decision and ending the 
response with `[END]` 

In my messages, responses from the plaintiff's counsel start with `PC:` and responses from defendant's counsel start 
with `DC:`.
Do not generate responses for the counsels.
"""

costs_system_template = """You are an assistant registrar in the Supreme Court of Singapore.
Your objective is to render a decision on costs for this application.

The applicable principles regarding costs are:   
The Court must order the costs of any proceedings in favour of a successful party, except when it appears to the Court 
that in the circumstances of the case some other order should be made as to the whole or any part of the costs.

If the Plaintiff is successful, the following shall apply:
* Costs of $1000 should be awarded to Plaintiff.
* If Plaintiff's arguments did not assist the court, the Plaintiff's costs should be reduced by the extent it did not.
* If Defendant's arguments were instrumental to the decision of the court, no costs may be awarded.

If the Defendant is successful, the following shall apply:
* Costs of $1000 should be awarded to Defendant.
* If Defendant's arguments did not assist the court, the Defendant's costs should be reduced by the extent it did not.
* If Plaintiff's arguments were instrumental to the decision of the court, no costs may be awarded.

Let's think step by step.
"""

costs_user_template = """This is a transcript of the arguments in the hearing:
{transcript}

This is the decision made by the court:
{decision}
"""