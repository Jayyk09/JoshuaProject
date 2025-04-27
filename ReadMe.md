## Joshua Project RAG Based Bot

#Datasets Information

CSV File Shapes:
--------------------------------------------------
Database: Joshua Project People Group Data
File: AllLanguageListing.csv
  Rows: 7148
  Columns: 14
  Column names: ['ROL3', 'Language', 'NbrPGICs', 'JPScale', 'LeastReached', 'BibleStatus', 'BibleYear', 'NTYear', 'PortionsYear', 'JF', 'AudioRecordings', 'YouVersion_ID', 'RLG3', 'PrimaryReligion']
--------------------------------------------------
Database: Joshua Project People Group Data
File: PeopleCtryLangListing.csv
  Rows: 46490
  Columns: 8
  Column names: ['PeopleID3', 'ROG3', 'ROL3', 'ROL4', 'Language', 'LanguageDialect', 'LanguageRank', 'Speakers']
--------------------------------------------------
Database: Type,PEID,ROP3,PeopleID3,ROG3,Ctry,JPPeopleGroup,JPPopulation,JPIndigenous,JPROL3,JPPrimaryLanguage,JPRLG3,JPPrimaryReligion,JPScale,JPLeastReached,JP%ChristianAdherent,JP%Evangelical,CPPIPeopleGroup, CPPIPopulation ,CPPIROL,CPPIPrimaryLanguage,CPPIPrimaryReligion,CPPIGSEC,CPPIEvangelicalEngagement
File: jp-cppi-cross-reference.csv
  Rows: 19373
  Columns: 24
  Column names: ['1', '23947', '100096', '19409', 'AF', 'Afghanistan', 'Afshari', '16000', 'N', 'azb', 'Azerbaijani, South', '6', 'Islam', '1.1', 'Y', '0.04', '0.04.1', 'Afshari.1', ' 13,500 ', 'azb.1', 'South Azerbaijani', 'Islam - Sunni', '1.2', 'Unengaged']
--------------------------------------------------
Database: Joshua Project People Group Data
File: AllCountriesListing.csv
  Rows: 253
  Columns: 23
  Column names: ['ROG3', 'ISO3', 'ISO2', 'Ctry', 'PoplPeoples', 'CntPeoples', 'CntPeoplesLR', 'PoplPeoplesLR', 'JPScaleCtry', 'ROL3OfficialLanguage', 'OfficialLang', 'RLG3Primary', 'ReligionPrimary', 'PercentChristianity', 'PercentEvangelical', '10_40Window', 'ROG2', 'Continent', 'RegionCode', 'RegionName', 'PercentUrbanized', 'LiteracyRate', 'WorkersNeeded']
--------------------------------------------------
Database: Joshua Project People Group Data
File: AllPeoplesAcrossCountries.csv
  Rows: 10379
  Columns: 25
  Column names: ['PeopleID3', 'PeopName', 'PeopleID1', 'AffinityBloc', 'PeopleID2', 'PeopleCluster', 'ROP3', 'ROP25', 'ROP25Name', 'JPScalePGAC', 'PopulationPGAC', 'LeastReachedPGAC', 'FrontierPGAC', 'CntPGIC', 'CntUPG', 'CntFPG', 'ROG3Largest', 'CtryLargest', 'ROL3PGAC', 'PrimaryLanguagePGAC', 'RLG3PGAC', 'PrimaryReligionPGAC', 'PercentChristianPGAC', 'PercentEvangelicalPGAC', 'PeopleID3General']
--------------------------------------------------
Database: Joshua Project People Group Data
File: FieldDefinitions.csv
  Rows: 247
  Columns: 4
  Column names: ['TableName', 'FieldName', 'FieldDescription', 'FieldType']
--------------------------------------------------
Database: Joshua Project People Group Data
File: UnreachedPeoplesByCountry.csv
  Rows: 7259
  Columns: 33
  Column names: ['ROG3', 'Ctry', 'PeopleID3', 'ROP3', 'PeopNameAcrossCountries', 'PeopNameInCountry', 'Population', 'JPScale', 'LeastReached', 'ROL3', 'PrimaryLanguageName', 'BibleStatus', 'RLG3', 'PrimaryReligion', 'PercentAdherents', 'PercentEvangelical', 'PeopleID1', 'ROP1', 'AffinityBloc', 'PeopleID2', 'ROP2', 'PeopleCluster', 'CountOfCountries', 'RegionCode', 'RegionName', 'ROG2', 'Continent', '10_40Window', 'IndigenousCode', 'WorkersNeeded', 'Frontier', 'Latitude', 'Longitude']
--------------------------------------------------
Database: Joshua Project People Group Data
File: AllPeoplesInCountry.csv
  Rows: 17366
  Columns: 33
  Column names: ['ROG3', 'Ctry', 'PeopleID3', 'ROP3', 'PeopNameAcrossCountries', 'PeopNameInCountry', 'Population', 'JPScale', 'LeastReached', 'ROL3', 'PrimaryLanguageName', 'BibleStatus', 'RLG3', 'PrimaryReligion', 'PercentAdherents', 'PercentEvangelical', 'PeopleID1', 'ROP1', 'AffinityBloc', 'PeopleID2', 'ROP2', 'PeopleCluster', 'CountOfCountries', 'RegionCode', 'RegionName', 'ROG2', 'Continent', '10_40Window', 'IndigenousCode', 'WorkersNeeded', 'Frontier', 'Latitude', 'Longitude']
--------------------------------------------------