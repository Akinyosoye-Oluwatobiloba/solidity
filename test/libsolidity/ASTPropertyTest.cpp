/*
	This file is part of solidity.

	solidity is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	solidity is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with solidity.  If not, see <http://www.gnu.org/licenses/>.
*/
// SPDX-License-Identifier: GPL-3.0

#include <test/libsolidity/ASTPropertyTest.h>
#include <test/Common.h>

#include <libsolidity/ast/ASTJsonExporter.h>
#include <libsolidity/interface/CompilerStack.h>

#include <liblangutil/Common.h>

#include <boost/algorithm/string.hpp>
#include <boost/throw_exception.hpp>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/split.hpp>

#include <queue>

using namespace solidity::langutil;
using namespace solidity::frontend;
using namespace solidity::frontend::test;
using namespace solidity;
using namespace std;

ASTPropertyTest::ASTPropertyTest(string const& _filename):
	TestCase(_filename)
{
	if (!boost::algorithm::ends_with(_filename, ".sol"))
		BOOST_THROW_EXCEPTION(runtime_error("Not a Solidity file: \"" + _filename + "\"."));

	m_source = m_reader.source();
	readExpectations();
}

void ASTPropertyTest::generateTestCaseValues(string& _values, bool _obtained)
{
	_values.clear();
	for (auto testId: m_expectationsSequence)
	{
		soltestAssert(m_testCases.count(testId) > 0);
		_values +=
			testId +
			": " +
			(_obtained ? m_testCases[testId].obtainedValue : m_testCases[testId].expectedValue)
			+ "\n";
	}
}

vector<StringPair> ASTPropertyTest::readKeyValuePairs(string const& _input)
{
	vector<StringPair> result;
	for (string const& line: _input | ranges::views::split('\n') | ranges::to<vector<string>>)
	{
		if (line.empty())
			continue;

		vector<string> pair;
		for (string const& element: line | ranges::views::split(':') | ranges::to<vector<string>>)
		{
			soltestAssert(ranges::all_of(boost::trim_copy(element), [](char c) { return isprint(c); }));
			pair.emplace_back(boost::trim_copy(element));
		}

		soltestAssert(pair.size() == 2);
		result.emplace_back(pair[0], pair[1]);
	}
	return result;
}

void ASTPropertyTest::readExpectations()
{
	for (auto const& [testId, testExpectation]: readKeyValuePairs(m_reader.simpleExpectations()))
	{
		m_testCases.emplace(testId, ASTPropertyTestCase{testId, "", testExpectation, ""});
		m_expectationsSequence.push_back(testId);
	}
	generateTestCaseValues(m_expectation, false);
}

optional<Json::Value> ASTPropertyTest::findNode(Json::Value const& _root, string_view const& _property)
{
	if (!_property.empty())
	{
		string subNode = string(_property.substr(0, _property.find_first_of('.')));
		if (subNode != _property)
			return findNode(_root[subNode], _property.substr(_property.find_first_of('.') + 1));
		else if (_root.isMember(subNode))
			return make_optional(_root[subNode]);
	}
	return {};
}

void ASTPropertyTest::readTestedProperties(Json::Value const& _astJson)
{
	queue<Json::Value> nodesToVisit;
	nodesToVisit.push(_astJson);
	string documentation = "documentation";
	string testCaseLine;

	while(!nodesToVisit.empty())
	{
		auto& node = nodesToVisit.front();
		if (node.isObject())
		{
			for (auto memberName: node.getMemberNames())
				if (memberName == documentation)
				{
					auto docNode = node[documentation];
					testCaseLine = docNode.isObject() ?
						docNode["text"].asString() :
						docNode.asString();

					soltestAssert(!testCaseLine.empty());
					auto pairs = readKeyValuePairs(testCaseLine);
					soltestAssert(!pairs.empty());
					auto [testId, testedProperty] = pairs[0];
					m_testCases[testId].property = testedProperty;
					auto propertyNode = findNode(node, testedProperty);
					soltestAssert(propertyNode.has_value(), "Could not find "s + testedProperty);
					soltestAssert(!propertyNode->isObject());
					m_testCases[testId].obtainedValue = propertyNode->asString();
				}
				else
					nodesToVisit.push(node[memberName]);
		}
		else if (node.isArray())
			for (auto&& member: node)
				nodesToVisit.push(member);

		nodesToVisit.pop();
	}
	generateTestCaseValues(m_obtainedResult, true);
}

TestCase::TestResult ASTPropertyTest::run(ostream& _stream, string const& _linePrefix, bool const _formatted)
{
	CompilerStack compiler;

	compiler.setSources({{
		"A",
		"pragma solidity >=0.0;\n// SPDX-License-Identifier: GPL-3.0\n" + m_source
	}});
	compiler.setEVMVersion(solidity::test::CommonOptions::get().evmVersion());
	compiler.setOptimiserSettings(solidity::test::CommonOptions::get().optimize);
	if (!compiler.parseAndAnalyze())
		BOOST_THROW_EXCEPTION(runtime_error("Parsing contract failed"));

	Json::Value astJson = ASTJsonExporter(compiler.state()).toJson(compiler.ast("A"));
	soltestAssert(astJson);

	readTestedProperties(astJson);

	return checkResult(_stream, _linePrefix, _formatted);
}
